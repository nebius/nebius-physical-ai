"""Enforce the private, fail-closed MolmoAct offline inference contract."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import numbers
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import stat
import sys
import sysconfig
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence
import unicodedata
from urllib.parse import urlsplit


RUNTIME_LOCK_PATH = Path(__file__).with_name("molmoact_runtime_lock.json")
PAYLOAD_DOMAIN = b"npa.molmoact.inference-payload.v1\x00"
CONSUMER_MANIFEST_DOMAIN = b"npa.molmoact.consumer-view-manifest.v1\x00"
GATE_A_RECEIPT_DOMAIN = b"npa.molmoact.gate-a-environment-receipt.v1\x00"
GATE_A_RECORD_DOMAIN = b"npa.molmoact.gate-a-record-rows.v1\x00"

_ERROR_CODES = frozenset(
    {
        "MOLMOACT_DECISION_INVALID",
        "MOLMOACT_DECISION_EXPIRED",
        "MOLMOACT_DECISION_BINDING",
        "MOLMOACT_IMAGE_DECLARATION",
        "MOLMOACT_EGRESS_EVIDENCE",
        "MOLMOACT_LOCK_INVALID",
        "MOLMOACT_SNAPSHOT_SHAPE",
        "MOLMOACT_SNAPSHOT_SIZE",
        "MOLMOACT_GIT_BLOB_IDENTITY",
        "MOLMOACT_LFS_IDENTITY",
        "MOLMOACT_QWEN_WEIGHT_PRESENT",
        "MOLMOACT_INPUT_IDENTITY",
        "MOLMOACT_DESTINATION",
        "MOLMOACT_HEAVY_PREIMPORTED",
        "MOLMOACT_OFFLINE_STATE",
        "MOLMOACT_HUB_KERNELS",
        "MOLMOACT_NETWORK_ATTEMPT",
        "MOLMOACT_RUNTIME_LOAD",
        "MOLMOACT_GENERATION",
        "MOLMOACT_DEPTH_GRAMMAR",
        "MOLMOACT_TRACE_GRAMMAR",
        "MOLMOACT_ACTION_TOKEN_GRAMMAR",
        "MOLMOACT_ACTION_PARSE",
        "MOLMOACT_OUTPUT_COLLISION",
        "MOLMOACT_OUTPUT_DURABILITY",
        "MOLMOACT_OUTPUT_READBACK",
    }
)

_MODEL_REPOSITORY = "allenai/MolmoAct-7B-O-0812"
_MODEL_REVISION = "b614d9041b68b1e79bd1337a0308ad74bd5df8ae"
_QWEN_REPOSITORY = "Qwen/Qwen2-7B"
_QWEN_REVISION = "453ed1575b739b5b03ce3758b23befdb0967f40e"
_INPUT_REPOSITORY = "allenai/MolmoAct-7B-D-0812"
_INPUT_REVISION = "936bcc40a2d9bda45cab41d11a06fbe2782ae360"
_PROMPT_SHA256 = "29a1cc6b4d29bf5c3cb6e9654d2ca5dac99ca91126325f3ae8f6a31966a346f8"
_OPERATION_SHA256 = "3a016011b6dc3448c0ef41c4e1246502f6557415210ab160a45078a2148696c6"
_QWEN_CLOSURE_SHA256 = (
    "beb6a0e21024812e339920bae9513d3ed6b6f4656327eec5b66b16d9e9f4732e"
)
_CONSUMER_MANIFEST_SHA256 = (
    "83508fd23ba77c29b9df1d48c56c18eb80aa7f47869434f2d3acc93443f0c3b9"
)
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_MODEL_FILE_FACTS = (
    (
        "added_tokens.json",
        "tokenizer",
        18676,
        "git_blob_sha1",
        "5f257ca6212da09bb31660b2689066c5864f622b",
    ),
    (
        "chat_template.jinja",
        "prompt template",
        676,
        "git_blob_sha1",
        "90161ea1b72db2c989222c796471ace0698185c8",
    ),
    (
        "config.json",
        "model config",
        3755,
        "git_blob_sha1",
        "ff31225cd6fecf4ebd321d1d755a81c8e9c1bc40",
    ),
    (
        "configuration_molmoact.py",
        "executable remote code",
        12658,
        "git_blob_sha1",
        "35fc692604160ce1a526ce033d7df7989d99c2d2",
    ),
    (
        "generation_config.json",
        "generation config",
        117,
        "git_blob_sha1",
        "9f7adfa7d7f3bdc2fc54f884e44535b6066b5bcb",
    ),
    (
        "image_processing_molmoact.py",
        "executable remote code",
        36696,
        "git_blob_sha1",
        "b9d63d0f441483d4a0e624797b85e722601fc2ea",
    ),
    (
        "merges.txt",
        "processor tokenizer",
        916646,
        "git_blob_sha1",
        "354558edcdbd64ca7abd407b8be3d5d09d39d781",
    ),
    (
        "model-00001-of-00007.safetensors",
        "weights",
        4892921912,
        "lfs_sha256",
        "22b149604bdd0e6d5a1eab6a7764e9bc001ba2eaefd1f6506761746069027893",
    ),
    (
        "model-00002-of-00007.safetensors",
        "weights",
        4857403152,
        "lfs_sha256",
        "4308afe5a8c8c3e9e17cae20b9a4ef5ec81bf9f959511047855b2e9ae617784a",
    ),
    (
        "model-00003-of-00007.safetensors",
        "weights",
        4857403200,
        "lfs_sha256",
        "6564f9fdaaa424eaadfb33cf7e1a10ad9c6c7a0af1b90085fe8aae833ed52792",
    ),
    (
        "model-00004-of-00007.safetensors",
        "weights",
        4857403200,
        "lfs_sha256",
        "734aa8101f1689d7e41f8cbfaff539248712aa879d21a03578d37839b6107c9d",
    ),
    (
        "model-00005-of-00007.safetensors",
        "weights",
        4857403200,
        "lfs_sha256",
        "5b024919352d9a1adddfe53bf0accf25a86ceb4168fd9c95d5ff434df4c4bad5",
    ),
    (
        "model-00006-of-00007.safetensors",
        "weights",
        4651517480,
        "lfs_sha256",
        "90ae62928101ae357c6c7eb3c89bf9c9919dc9e5056a4f8b4be9aa16b343f051",
    ),
    (
        "model-00007-of-00007.safetensors",
        "weights",
        1652555904,
        "lfs_sha256",
        "1656295cc35ed895cb50a7557fc13c5619718f01c1df02f32e80a4625011bf9e",
    ),
    (
        "model.safetensors.index.json",
        "weight index",
        67306,
        "git_blob_sha1",
        "1df0691a8bc4af92f69bdc069be528a2dd2a57d8",
    ),
    (
        "model.yaml",
        "upstream model metadata",
        4790,
        "git_blob_sha1",
        "a0ded64e732b7145f031e199334056cfd6a07d45",
    ),
    (
        "modeling_molmoact.py",
        "executable remote code and parsers",
        93310,
        "git_blob_sha1",
        "4a4b0f9d69bfa90b4982773d925776fe5ca6260d",
    ),
    (
        "preprocessor_config.json",
        "processor config",
        633,
        "git_blob_sha1",
        "b53930c9b1d5e93b99b8850db83823f5f70abb07",
    ),
    (
        "processing_molmoact.py",
        "executable remote code",
        17773,
        "git_blob_sha1",
        "5b6b5facb473d5d2fa488d997b619935e1f7e600",
    ),
    (
        "processor_config.json",
        "processor config",
        369,
        "git_blob_sha1",
        "ae98dd64c6df616f9d29a3938141f320f318a4a5",
    ),
    (
        "special_tokens_map.json",
        "processor tokenizer",
        71589,
        "git_blob_sha1",
        "0b2daf0d484cd7fe3a3e2d8d0e33516d0bdcccf7",
    ),
    (
        "tokenizer.json",
        "processor tokenizer",
        7252346,
        "git_blob_sha1",
        "b760f50efb9df0d99a9ceacaf03bea46210cd3bc",
    ),
    (
        "tokenizer_config.json",
        "processor tokenizer",
        126514,
        "git_blob_sha1",
        "06da6dffe7393c7200020b670c2be7f19e70adf4",
    ),
    (
        "vocab.json",
        "processor tokenizer",
        1611056,
        "git_blob_sha1",
        "51135344eec01a62fc4deaca39c72ac08f5b9709",
    ),
)
_QWEN_FILE_FACTS = (
    (
        "LICENSE",
        "retained_notice",
        11343,
        "git_blob_sha1",
        "6634c8cc3133b3848ec74b9f275acaaa1ea618ab",
    ),
    (
        "merges.txt",
        "tokenizer_merges",
        1671839,
        "git_blob_sha1",
        "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0",
    ),
    (
        "tokenizer_config.json",
        "tokenizer_init_metadata",
        1289,
        "git_blob_sha1",
        "f4b55f917af273d0dc98b67ec249f6445dd385f5",
    ),
    (
        "vocab.json",
        "tokenizer_vocabulary",
        2776833,
        "git_blob_sha1",
        "4783fe10ac3adce15ac8f358ef5462739852c569",
    ),
)
_INPUT_FILE_FACTS = (
    (
        "example_1.png",
        "side view, first image",
        352341,
        "lfs_sha256",
        "19253fd1d5286ac6b1e72fa4c71c8918b318c2a8c9358909ed793107e90a70d2",
    ),
    (
        "example_2.png",
        "wrist view, second image",
        377478,
        "lfs_sha256",
        "95ddc7159ec1657f667709e805ebec28ad22e42bd10e3efcf7186b9999434d8b",
    ),
)
_CONSUMER_DIRECTORIES = [
    ".",
    "hf_modules",
    "hf_modules/transformers_modules",
    "hf_modules/transformers_modules/molmoact_model",
    "inputs",
    "molmoact_model",
    "qwen_tokenizer",
]
_CACHE_COPY_NAMES = (
    "configuration_molmoact.py",
    "image_processing_molmoact.py",
    "modeling_molmoact.py",
    "processing_molmoact.py",
)
_TERM_IDS = (
    "model_card_and_terms",
    "model_executable_remote_code",
    "input_card_and_images",
    "qwen_license_and_tokenizer",
    "apache_2_0",
    "ai2_responsible_use_guidelines",
)
_HEAVY_PREFIXES = (
    "PIL",
    "numpy",
    "torch",
    "transformers",
    "huggingface_hub",
    "kernels",
)
_QWEN_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".index.json",
)
_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_DECISION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_OCI_PATTERN = re.compile(
    r"(?P<registry>[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?)/"
    r"(?P<repository>[a-z0-9]+(?:[._-][a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*)"
    r"@sha256:[0-9a-f]{64}"
)


class MolmoActInferenceError(RuntimeError):
    """Report one stable fail-closed MolmoAct contract error.

    Args:
        code: One frozen contract error code.
        detail: A bounded diagnostic without payload contents.
        context: Structured publication-state details when applicable.
    Returns:
        None.
    Raises:
        ValueError: The caller uses an unknown error code.
    """

    def __init__(self, code: str, detail: str, **context: Any) -> None:
        if code not in _ERROR_CODES:
            raise ValueError(f"unknown MolmoAct error code: {code}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.context = dict(context)

    def as_dict(self) -> dict[str, Any]:
        """Return a canonicalizable diagnostic object.

        Args:
            None.
        Returns:
            The stable error code, detail, and structured context.
        Raises:
            None.
        """

        return {"code": self.code, "detail": self.detail, **self.context}


@dataclass(frozen=True)
class ValidatedMachineRecord:
    """Hold strict record bytes, parsed data, and the complete-file digest.

    Args:
        data: Strict parsed top-level object.
        raw_bytes: Complete original file bytes.
        sha256: Complete-file lowercase SHA-256.
        path: Original absolute path when the record was path-backed.
    Returns:
        None.
    Raises:
        None.
    """

    data: dict[str, Any]
    raw_bytes: bytes
    sha256: str
    path: str | None = None


@dataclass(frozen=True)
class MountObservation:
    """Describe the locally observed read-only consumer-view mount.

    Args:
        mount_id: Decimal Linux mount ID.
        st_dev: Consumer root ``st_dev``.
        major: Mountinfo major device number.
        minor: Mountinfo minor device number.
        mount_options: Decoded mount options.
        superblock_options: Decoded superblock options.
        statvfs_read_only: Whether ``ST_RDONLY`` was observed.
    Returns:
        None.
    Raises:
        None.
    """

    mount_id: int
    st_dev: int
    major: int
    minor: int
    mount_options: tuple[str, ...]
    superblock_options: tuple[str, ...]
    statvfs_read_only: bool


@dataclass
class VerifiedSnapshot:
    """Retain verified snapshot descriptors until all consumers finish.

    Args:
        root: Anchored lexical snapshot root.
        records: Ordered verified lock and identity records.
        descriptors: Ordered-name retained no-follow descriptors.
    Returns:
        None.
    Raises:
        None.
    """

    root: str
    records: list[dict[str, Any]]
    descriptors: dict[str, int]

    def close(self) -> None:
        """Close all retained verified file descriptors.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """

        for descriptor in self.descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.descriptors.clear()


@dataclass(frozen=True)
class _AnchoredDirectory:
    path: str
    descriptor: int
    device: int
    inode: int


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one value with the contract's canonical JSON profile.

    Args:
        value: A JSON-compatible value.
    Returns:
        UTF-8 JSON without a trailing newline.
    Raises:
        TypeError: The value is not JSON-compatible.
        ValueError: The value contains a nonfinite number.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def git_blob_sha1(data: bytes) -> str:
    """Hash bytes using Git's blob-object framing.

    Args:
        data: Complete file bytes.
    Returns:
        The lowercase Git blob SHA-1.
    Raises:
        TypeError: The input is not bytes.
    """

    if not isinstance(data, bytes):
        raise TypeError("Git blob input must be bytes")
    header = b"blob " + str(len(data)).encode("ascii") + b"\x00"
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _strict_json_value(raw_bytes: bytes) -> Any:
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError("UTF-8 BOM is forbidden")
    text = raw_bytes.decode("utf-8", errors="strict")
    if not text or text[0].isspace() or text[-1].isspace():
        raise ValueError("leading or trailing JSON whitespace is forbidden")
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )


def _strict_json_object(raw_bytes: bytes) -> dict[str, Any]:
    value = _strict_json_value(raw_bytes)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _validate_component(component: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or unicodedata.normalize("NFC", component) != component
        or any(ord(character) < 32 or ord(character) == 127 for character in component)
    ):
        raise ValueError("invalid path component")
    component.encode("utf-8", errors="strict")


def _absolute_parts(path: str | os.PathLike[str]) -> tuple[str, list[str]]:
    value = os.fspath(path)
    if not isinstance(value, str) or not value.startswith("/") or value == "/":
        raise ValueError("path must be an absolute non-root path")
    if value.endswith("/") or "//" in value:
        raise ValueError("path contains an empty component")
    parts = value[1:].split("/")
    for part in parts:
        _validate_component(part)
    return value, parts


def _anchor_directory(path: str | os.PathLike[str]) -> _AnchoredDirectory:
    if os.fspath(path) == "/":
        descriptor = os.open(
            "/",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        details = os.fstat(descriptor)
        return _AnchoredDirectory("/", descriptor, details.st_dev, details.st_ino)
    value, parts = _absolute_parts(path)
    descriptor = os.open(
        "/",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        for part in parts:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        details = os.fstat(descriptor)
        if not stat.S_ISDIR(details.st_mode):
            raise ValueError("anchored path is not a directory")
        return _AnchoredDirectory(value, descriptor, details.st_dev, details.st_ino)
    except BaseException:
        os.close(descriptor)
        raise


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _read_absolute_regular(
    path: str | os.PathLike[str],
    *,
    private: bool = False,
) -> tuple[bytes, os.stat_result]:
    value, parts = _absolute_parts(path)
    parent_path = "/" + "/".join(parts[:-1])
    parent = _anchor_directory(parent_path)
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent.descriptor,
        )
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise ValueError("record is not a regular file")
            if private and details.st_mode & 0o022:
                raise ValueError("record is group/world writable")
            return _read_descriptor(descriptor), details
        finally:
            os.close(descriptor)
    finally:
        os.close(parent.descriptor)


def read_strict_json(
    source: bytes | str | os.PathLike[str],
    *,
    error_code: str,
    private: bool = False,
) -> ValidatedMachineRecord:
    """Read one duplicate-free, finite, exact UTF-8 JSON object.

    Args:
        source: Complete JSON bytes or an absolute no-symlink path.
        error_code: Stable code used for malformed input.
        private: Require a path-backed record not be group/world writable.
    Returns:
        Parsed data, original bytes, and complete-file SHA-256.
    Raises:
        MolmoActInferenceError: The source or JSON is invalid.
    """

    try:
        if isinstance(source, bytes):
            raw_bytes = source
            path = None
        else:
            raw_bytes, _ = _read_absolute_regular(source, private=private)
            path = os.fspath(source)
        data = _strict_json_object(raw_bytes)
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            error_code, "strict JSON record is invalid"
        ) from exc
    return ValidatedMachineRecord(
        data=data,
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=path,
    )


def _file_objects(
    facts: Sequence[tuple[str, str, int, str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "role": role,
            "size": size,
            "identity_kind": identity_kind,
            "identity": identity,
        }
        for path, role, size, identity_kind, identity in facts
    ]


def _require_keys(value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("object keys do not match the frozen schema")
    return value


def _require_lower_digest(value: Any, length: int) -> str:
    pattern = rf"[0-9a-f]{{{length}}}"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError("digest has the wrong lexical form")
    return value


def _validate_file_table(
    value: Any,
    facts: Sequence[tuple[str, str, int, str, str]],
) -> None:
    expected = _file_objects(facts)
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError("snapshot file table has the wrong length")
    seen: set[str] = set()
    folded: set[str] = set()
    for item in value:
        item = _require_keys(
            item, {"path", "role", "size", "identity_kind", "identity"}
        )
        if (
            not isinstance(item["path"], str)
            or not isinstance(item["role"], str)
            or type(item["size"]) is not int
            or item["identity_kind"] not in {"git_blob_sha1", "lfs_sha256"}
        ):
            raise ValueError("snapshot file field has the wrong type")
        _validate_component(item["path"])
        digest_length = 40 if item["identity_kind"] == "git_blob_sha1" else 64
        _require_lower_digest(item["identity"], digest_length)
        if item["path"] in seen or item["path"].casefold() in folded:
            raise ValueError("snapshot paths are duplicate or case-colliding")
        seen.add(item["path"])
        folded.add(item["path"].casefold())
    if value != expected:
        raise ValueError("snapshot file table differs from the frozen vector")


def _validate_snapshot_lock(
    value: Any,
    *,
    repository: str,
    revision: str,
    facts: Sequence[tuple[str, str, int, str, str]],
) -> None:
    item = _require_keys(value, {"repository", "revision", "files"})
    if item["repository"] != repository or item["revision"] != revision:
        raise ValueError("snapshot repository or revision differs")
    _require_lower_digest(item["revision"], 40)
    _validate_file_table(item["files"], facts)


def _operation_projection(lock: Mapping[str, Any]) -> dict[str, Any]:
    inputs = lock["input_snapshot"]
    return {
        "operation": "one_shot_offline_inference",
        "instruction": lock["operation"]["instruction"],
        "prompt_sha256": lock["operation"]["prompt_sha256"],
        "model_repository": lock["model_snapshot"]["repository"],
        "model_revision": lock["model_snapshot"]["revision"],
        "qwen_repository": lock["qwen_snapshot"]["repository"],
        "qwen_revision": lock["qwen_snapshot"]["revision"],
        "input_repository": inputs["repository"],
        "input_revision": inputs["revision"],
        "input_files": [
            {"path": item["path"], "sha256": item["identity"]}
            for item in inputs["files"]
        ],
    }


def _qwen_projection(lock: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = lock["qwen_snapshot"]
    files = [
        {
            "identity": item["identity"],
            "identity_kind": item["identity_kind"],
            "path": item["path"],
            "role": item["role"],
            "size": item["size"],
        }
        for item in snapshot["files"]
    ]
    return {
        "files": sorted(files, key=lambda item: item["path"]),
        "repository": snapshot["repository"],
        "revision": snapshot["revision"],
    }


def _validate_operation(lock: Mapping[str, Any]) -> None:
    operation = _require_keys(
        lock["operation"],
        {
            "instruction",
            "prompt",
            "prompt_sha256",
            "operation_canonical_json",
            "operation_sha256",
            "qwen_closure_canonical_json",
            "qwen_closure_sha256",
            "action_unnorm_key",
            "expected_action_dim",
            "depth_tokens",
            "generation",
        },
    )
    if operation["instruction"] != "close the box":
        raise ValueError("instruction differs from the frozen operation")
    if not isinstance(operation["prompt"], str):
        raise ValueError("prompt must be a string")
    if (
        hashlib.sha256(operation["prompt"].encode("utf-8")).hexdigest()
        != _PROMPT_SHA256
    ):
        raise ValueError("prompt digest differs from the frozen operation")
    if operation["prompt_sha256"] != _PROMPT_SHA256:
        raise ValueError("prompt digest binding differs")
    _validate_operation_vectors(lock, operation)
    _validate_operation_scalars(operation)


def _validate_operation_vectors(
    lock: Mapping[str, Any], operation: Mapping[str, Any]
) -> None:
    operation_bytes = canonical_json_bytes(_operation_projection(lock))
    qwen_bytes = canonical_json_bytes(_qwen_projection(lock))
    for literal in (
        operation["operation_canonical_json"],
        operation["qwen_closure_canonical_json"],
    ):
        if not isinstance(literal, str):
            raise ValueError("canonical vector must be a string")
        _strict_json_value(literal.encode("utf-8"))
    if operation_bytes.decode("utf-8") != operation["operation_canonical_json"]:
        raise ValueError("operation canonical vector differs")
    if hashlib.sha256(operation_bytes).hexdigest() != _OPERATION_SHA256:
        raise ValueError("operation canonical digest differs")
    if operation["operation_sha256"] != _OPERATION_SHA256:
        raise ValueError("operation digest binding differs")
    if qwen_bytes.decode("utf-8") != operation["qwen_closure_canonical_json"]:
        raise ValueError("Qwen closure canonical vector differs")
    if hashlib.sha256(qwen_bytes).hexdigest() != _QWEN_CLOSURE_SHA256:
        raise ValueError("Qwen closure digest differs")
    if operation["qwen_closure_sha256"] != _QWEN_CLOSURE_SHA256:
        raise ValueError("Qwen closure digest binding differs")


def _validate_operation_scalars(operation: Mapping[str, Any]) -> None:
    depth = _require_keys(operation["depth_tokens"], {"count", "minimum", "maximum"})
    generation = _require_keys(
        operation["generation"],
        {
            "max_new_tokens",
            "do_sample",
            "num_beams",
            "num_return_sequences",
            "use_cache",
        },
    )
    if operation["action_unnorm_key"] != "molmoact":
        raise ValueError("action unnormalization key differs")
    if type(operation["expected_action_dim"]) is not int:
        raise ValueError("action dimension has the wrong type")
    if operation["expected_action_dim"] != 7:
        raise ValueError("action dimension differs")
    if any(type(depth[key]) is not int for key in depth):
        raise ValueError("depth bounds have the wrong type")
    if depth != {"count": 100, "minimum": 1, "maximum": 128}:
        raise ValueError("depth bounds differ")
    expected = {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "use_cache": True,
    }
    if any(type(generation[key]) is not type(expected[key]) for key in expected):
        raise ValueError("generation field has the wrong exact type")
    if generation != expected:
        raise ValueError("generation settings differ")


def derive_consumer_manifest(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the complete 37-file consumer manifest from strict lock facts.

    Args:
        lock: A validated or candidate runtime-lock object.
    Returns:
        The canonical manifest projection without digest metadata.
    Raises:
        KeyError: Required lock facts are absent.
    """

    files: list[dict[str, Any]] = []
    for prefix, snapshot in (
        ("molmoact_model", lock["model_snapshot"]),
        ("inputs", lock["input_snapshot"]),
        ("qwen_tokenizer", lock["qwen_snapshot"]),
    ):
        for item in snapshot["files"]:
            path = f"{prefix}/{item['path']}"
            files.append(
                {
                    "identity": item["identity"],
                    "identity_kind": item["identity_kind"],
                    "kind": "payload",
                    "path": path,
                    "role": item["role"],
                    "size": item["size"],
                    "source_path": path,
                }
            )
    files.extend(_generated_cache_entries(lock))
    return {
        "directories": list(_CONSUMER_DIRECTORIES),
        "files": sorted(files, key=lambda item: item["path"]),
        "schema": "npa.molmoact.consumer-view-manifest.v1",
    }


def _generated_cache_entries(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [
        {
            "identity": _EMPTY_SHA256,
            "identity_kind": "sha256",
            "kind": "generated_empty",
            "path": path,
            "role": "dynamic_cache_init",
            "size": 0,
            "source_path": "generated:empty",
        }
        for path in (
            "hf_modules/__init__.py",
            "hf_modules/transformers_modules/__init__.py",
            "hf_modules/transformers_modules/molmoact_model/__init__.py",
        )
    ]
    model_files = {item["path"]: item for item in lock["model_snapshot"]["files"]}
    for name in _CACHE_COPY_NAMES:
        source = model_files[name]
        entries.append(
            {
                "identity": source["identity"],
                "identity_kind": source["identity_kind"],
                "kind": "cache_copy",
                "path": f"hf_modules/transformers_modules/molmoact_model/{name}",
                "role": "dynamic_cache_copy",
                "size": source["size"],
                "source_path": f"molmoact_model/{name}",
            }
        )
    return entries


def _validate_manifest_entry(item: Any) -> None:
    item = _require_keys(
        item,
        {
            "identity",
            "identity_kind",
            "kind",
            "path",
            "role",
            "size",
            "source_path",
        },
    )
    if (
        any(not isinstance(item[key], str) for key in item if key != "size")
        or type(item["size"]) is not int
        or item["size"] < 0
    ):
        raise ValueError("manifest entry has the wrong exact type")
    if item["kind"] not in {"payload", "cache_copy", "generated_empty"}:
        raise ValueError("manifest entry kind is invalid")
    if item["identity_kind"] not in {"git_blob_sha1", "lfs_sha256", "sha256"}:
        raise ValueError("manifest identity kind is invalid")
    _validate_relative_path(item["path"])
    digest_length = 40 if item["identity_kind"] == "git_blob_sha1" else 64
    _require_lower_digest(item["identity"], digest_length)


def _validate_relative_path(path: str) -> None:
    if not isinstance(path, str) or path.startswith("/") or "\\" in path:
        raise ValueError("manifest path is not relative")
    for component in path.split("/"):
        _validate_component(component)


def _validate_consumer_layout(lock: Mapping[str, Any]) -> None:
    layout = _require_keys(
        lock["consumer_view_layout"],
        {"schema", "manifest_domain", "manifest_sha256", "directories", "files"},
    )
    expected = derive_consumer_manifest(lock)
    if layout["schema"] != "npa.molmoact.consumer-view-manifest.v1":
        raise ValueError("consumer manifest schema differs")
    if layout["manifest_domain"] != "npa.molmoact.consumer-view-manifest.v1\x00":
        raise ValueError("consumer manifest domain differs")
    if layout["directories"] != _CONSUMER_DIRECTORIES:
        raise ValueError("consumer directory vector differs")
    if not isinstance(layout["files"], list):
        raise ValueError("consumer manifest files are not an array")
    for item in layout["files"]:
        _validate_manifest_entry(item)
    paths = [item["path"] for item in layout["files"]]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("consumer file paths are not unique lexical order")
    if len({path.casefold() for path in paths}) != len(paths):
        raise ValueError("consumer file paths case-collide")
    if layout["files"] != expected["files"] or len(paths) != 37:
        raise ValueError("consumer manifest projection differs")
    projection = canonical_json_bytes(expected)
    if len(projection) != 9585:
        raise ValueError("consumer manifest canonical byte length differs")
    digest = hashlib.sha256(CONSUMER_MANIFEST_DOMAIN + projection).hexdigest()
    if digest != _CONSUMER_MANIFEST_SHA256:
        raise ValueError("consumer manifest digest differs")
    if layout["manifest_sha256"] != digest:
        raise ValueError("consumer manifest digest binding differs")


def validate_runtime_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every runtime-lock scalar and canonical vector exactly.

    Args:
        lock: Parsed candidate runtime-lock object.
    Returns:
        The same validated object as a plain dictionary.
    Raises:
        MolmoActInferenceError: Any key, type, value, order, or digest differs.
    """

    try:
        value = _require_keys(
            lock,
            {
                "schema",
                "transformers_version",
                "forbidden_distributions",
                "model_snapshot",
                "qwen_snapshot",
                "input_snapshot",
                "operation",
                "consumer_view_layout",
            },
        )
        if value["schema"] != "npa.molmoact.runtime-lock.v2":
            raise ValueError("runtime lock schema differs")
        if value["transformers_version"] != "4.52.3":
            raise ValueError("Transformers version differs")
        if value["forbidden_distributions"] != ["kernels"]:
            raise ValueError("forbidden distribution vector differs")
        _validate_snapshot_lock(
            value["model_snapshot"],
            repository=_MODEL_REPOSITORY,
            revision=_MODEL_REVISION,
            facts=_MODEL_FILE_FACTS,
        )
        _validate_snapshot_lock(
            value["qwen_snapshot"],
            repository=_QWEN_REPOSITORY,
            revision=_QWEN_REVISION,
            facts=_QWEN_FILE_FACTS,
        )
        _validate_snapshot_lock(
            value["input_snapshot"],
            repository=_INPUT_REPOSITORY,
            revision=_INPUT_REVISION,
            facts=_INPUT_FILE_FACTS,
        )
        _validate_operation(value)
        _validate_consumer_layout(value)
    except (KeyError, TypeError, UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_LOCK_INVALID", "runtime lock differs from the frozen contract"
        ) from exc
    return dict(value)


def load_runtime_lock(
    path: str | os.PathLike[str] | None = None,
) -> ValidatedMachineRecord:
    """Load and exactly validate the checked-in runtime lock.

    Args:
        path: Optional explicit strict lock path.
    Returns:
        Strict lock data, original bytes, and complete-file SHA-256.
    Raises:
        MolmoActInferenceError: The file or any lock fact is invalid.
    """

    source = RUNTIME_LOCK_PATH if path is None else path
    record = read_strict_json(source, error_code="MOLMOACT_LOCK_INVALID")
    validate_runtime_lock(record.data)
    return record


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError("timestamp is not exact UTC second form")
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return parsed.replace(tzinfo=timezone.utc)


def _utc_now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("current time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _require_nonempty(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("required string is empty")
    return value


def _validate_oci_digest(value: Any) -> str:
    if not isinstance(value, str) or _OCI_PATTERN.fullmatch(value) is None:
        raise ValueError("candidate image is not a canonical full OCI digest")
    registry = value.split("/", 1)[0]
    if "." not in registry and ":" not in registry and registry != "localhost":
        raise ValueError("candidate image lacks an explicit registry")
    return value


def _validate_decision_terms(value: Any) -> None:
    if not isinstance(value, list) or len(value) != len(_TERM_IDS):
        raise ValueError("terms array has the wrong length")
    if [item.get("id") if isinstance(item, dict) else None for item in value] != list(
        _TERM_IDS
    ):
        raise ValueError("terms IDs or order differ")
    references: set[str] = set()
    for item in value:
        item = _require_keys(
            item,
            {
                "id",
                "reference",
                "immutable_revision_or_sha256",
                "reviewed_at",
                "disposition",
            },
        )
        reference = _validate_review_reference(item["reference"])
        if reference in references:
            raise ValueError("term references are duplicate")
        references.add(reference)
        immutable_identity = _require_nonempty(item["immutable_revision_or_sha256"])
        if (
            _REVISION_PATTERN.fullmatch(immutable_identity) is None
            and _DIGEST_PATTERN.fullmatch(immutable_identity) is None
        ):
            raise ValueError("term identity is not an immutable revision or digest")
        _parse_timestamp(item["reviewed_at"])
        if item["disposition"] != "approved_for_this_bounded_operation":
            raise ValueError("term disposition differs")


def _validate_review_reference(value: Any) -> str:
    reference = _require_nonempty(value)
    if (
        reference != reference.strip()
        or unicodedata.normalize("NFC", reference) != reference
        or "\\" in reference
        or any(character.isspace() for character in reference)
        or any(ord(character) < 32 or ord(character) == 127 for character in reference)
    ):
        raise ValueError("term reference has invalid text")
    if "://" in reference:
        parsed = urlsplit(reference)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("term reference is not a canonical HTTPS URL")
        return reference
    if "/" not in reference:
        raise ValueError("term repository reference lacks a path separator")
    _validate_relative_path(reference)
    return reference


def _decision_fixed_bindings(
    decision: Mapping[str, Any], lock: Mapping[str, Any], lock_sha256: str
) -> None:
    expected_inputs = {
        "repository": _INPUT_REPOSITORY,
        "revision": _INPUT_REVISION,
        "files": [
            {"path": item["path"], "sha256": item["identity"]}
            for item in lock["input_snapshot"]["files"]
        ],
    }
    expected = {
        "runtime_lock_sha256": lock_sha256,
        "model": {
            "repository": _MODEL_REPOSITORY,
            "revision": _MODEL_REVISION,
            "executable_code_revision": _MODEL_REVISION,
        },
        "qwen": {
            "repository": _QWEN_REPOSITORY,
            "revision": _QWEN_REVISION,
            "closure_sha256": _QWEN_CLOSURE_SHA256,
        },
        "inputs": expected_inputs,
        "instruction": "close the box",
        "prompt_sha256": _PROMPT_SHA256,
        "operation_sha256": _OPERATION_SHA256,
    }
    if any(decision[key] != expected[key] for key in expected):
        raise ValueError("decision does not bind the frozen operation")


def _validate_decision_scalars(decision: Mapping[str, Any]) -> None:
    if decision["schema"] != "npa.molmoact.use-scope-decision.v1":
        raise ValueError("decision schema differs")
    fixed = {
        "status": "approved_for_bounded_research",
        "scope": "scientific_research_and_development",
        "operation": "one_shot_offline_inference",
        "maximum_generations": 1,
        "robot_actuation": False,
    }
    if any(type(decision[key]) is not type(value) for key, value in fixed.items()):
        raise ValueError("decision scalar has the wrong exact type")
    if any(decision[key] != value for key, value in fixed.items()):
        raise ValueError("decision scalar differs")
    if (
        not isinstance(decision["decision_id"], str)
        or _DECISION_ID_PATTERN.fullmatch(decision["decision_id"]) is None
    ):
        raise ValueError("decision ID is invalid")
    for key in (
        "runtime_lock_sha256",
        "prompt_sha256",
        "operation_sha256",
        "egress_isolation_receipt_sha256",
    ):
        _require_lower_digest(decision[key], 64)
    _validate_oci_digest(decision["declared_candidate_image"])
    _require_nonempty(decision["authored_by_assertion"])
    _require_nonempty(decision["reviewed_by_assertion"])


def _validate_decision_nested_types(decision: Mapping[str, Any]) -> None:
    _require_keys(
        decision["model"],
        {"repository", "revision", "executable_code_revision"},
    )
    _require_keys(decision["qwen"], {"repository", "revision", "closure_sha256"})
    inputs = _require_keys(decision["inputs"], {"repository", "revision", "files"})
    if not isinstance(inputs["files"], list) or len(inputs["files"]) != 2:
        raise ValueError("decision input array differs")
    paths: set[str] = set()
    for item in inputs["files"]:
        item = _require_keys(item, {"path", "sha256"})
        _validate_component(item["path"])
        _require_lower_digest(item["sha256"], 64)
        if item["path"] in paths:
            raise ValueError("decision input paths are duplicate")
        paths.add(item["path"])
    for revision in (
        decision["model"]["revision"],
        decision["model"]["executable_code_revision"],
        decision["qwen"]["revision"],
        inputs["revision"],
    ):
        if (
            not isinstance(revision, str)
            or _REVISION_PATTERN.fullmatch(revision) is None
        ):
            raise ValueError("decision revision is malformed")
    _require_lower_digest(decision["qwen"]["closure_sha256"], 64)


def validate_decision(
    source: bytes | str | os.PathLike[str],
    *,
    lock: Mapping[str, Any],
    lock_sha256: str,
    now: datetime | None = None,
) -> ValidatedMachineRecord:
    """Validate a strict external bounded-use decision and all lock bindings.

    Args:
        source: Complete bytes or an absolute no-symlink decision path.
        lock: Exactly validated runtime lock.
        lock_sha256: Complete checked-in lock digest.
        now: Optional aware UTC test clock.
    Returns:
        The strict validated decision record.
    Raises:
        MolmoActInferenceError: Syntax, expiry, or binding is invalid.
    """

    record = read_strict_json(source, error_code="MOLMOACT_DECISION_INVALID")
    try:
        decision = _require_keys(
            record.data,
            {
                "schema",
                "decision_id",
                "status",
                "scope",
                "operation",
                "maximum_generations",
                "robot_actuation",
                "issued_at",
                "expires_at",
                "runtime_lock_sha256",
                "model",
                "qwen",
                "inputs",
                "instruction",
                "prompt_sha256",
                "operation_sha256",
                "terms",
                "declared_candidate_image",
                "egress_isolation_receipt_sha256",
                "authored_by_assertion",
                "reviewed_by_assertion",
            },
        )
        _validate_decision_scalars(decision)
        _validate_decision_nested_types(decision)
        _validate_decision_terms(decision["terms"])
        issued = _parse_timestamp(decision["issued_at"])
        expires = _parse_timestamp(decision["expires_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_DECISION_INVALID", "decision record is invalid"
        ) from exc
    try:
        _decision_fixed_bindings(decision, lock, lock_sha256)
    except (KeyError, TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_DECISION_BINDING", "decision binding differs"
        ) from exc
    current = _utc_now(now)
    if issued > current or current >= expires:
        raise MolmoActInferenceError(
            "MOLMOACT_DECISION_EXPIRED", "decision is not current"
        )
    return record


def validate_image_declaration(value: str | None, decision: Mapping[str, Any]) -> str:
    """Require the launcher image declaration to equal the approved OCI digest.

    Args:
        value: The ``NPA_TASK_IMAGE`` value.
        decision: Validated decision data.
    Returns:
        The byte-identical declared candidate image.
    Raises:
        MolmoActInferenceError: The declaration is absent, malformed, or unequal.
    """

    try:
        candidate = _validate_oci_digest(value)
    except (TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_IMAGE_DECLARATION", "NPA_TASK_IMAGE is not a full OCI digest"
        ) from exc
    if candidate != decision["declared_candidate_image"]:
        raise MolmoActInferenceError(
            "MOLMOACT_IMAGE_DECLARATION", "NPA_TASK_IMAGE differs from the decision"
        )
    return candidate


def _validate_isolation_data(
    receipt: Mapping[str, Any],
    decision: Mapping[str, Any],
    lock_sha256: str,
    observation: MountObservation,
    now: datetime | None,
) -> None:
    receipt = _require_keys(
        receipt,
        {
            "schema",
            "runtime_lock_sha256",
            "decision_id",
            "declared_candidate_image",
            "scope",
            "enforcement_mechanism",
            "issuer_assertion",
            "issued_at",
            "expires_at",
            "process_egress_denied",
            "subprocess_egress_denied",
            "consumer_view_manifest_sha256",
            "consumer_view_mount_id",
            "consumer_view_st_dev",
            "consumer_view_major",
            "consumer_view_minor",
            "consumer_view_read_only",
            "consumer_view_remount_denied",
            "cap_sys_admin_absent",
            "exclusive_one_shot_process",
            "same_euid_concurrency_denied",
        },
    )
    _validate_isolation_scalars(receipt)
    expected = {
        "runtime_lock_sha256": lock_sha256,
        "decision_id": decision["decision_id"],
        "declared_candidate_image": decision["declared_candidate_image"],
        "scope": decision["scope"],
        "consumer_view_manifest_sha256": _CONSUMER_MANIFEST_SHA256,
        "consumer_view_mount_id": observation.mount_id,
        "consumer_view_st_dev": observation.st_dev,
        "consumer_view_major": observation.major,
        "consumer_view_minor": observation.minor,
    }
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError("isolation receipt binding differs")
    if not observation.statvfs_read_only:
        raise ValueError("consumer view statvfs is writable")
    if (
        "ro" not in observation.mount_options
        or "ro" not in observation.superblock_options
    ):
        raise ValueError("consumer view mount options are writable")
    issued = _parse_timestamp(receipt["issued_at"])
    expires = _parse_timestamp(receipt["expires_at"])
    current = _utc_now(now)
    if issued > current or current >= expires:
        raise ValueError("isolation receipt is not current")


def _validate_isolation_scalars(receipt: Mapping[str, Any]) -> None:
    if receipt["schema"] != "npa.runtime.egress-isolation-receipt.v2":
        raise ValueError("isolation receipt schema differs")
    true_fields = (
        "process_egress_denied",
        "subprocess_egress_denied",
        "consumer_view_read_only",
        "consumer_view_remount_denied",
        "cap_sys_admin_absent",
        "exclusive_one_shot_process",
        "same_euid_concurrency_denied",
    )
    if any(
        type(receipt[key]) is not bool or receipt[key] is not True
        for key in true_fields
    ):
        raise ValueError("isolation control is not exact true")
    integer_fields = (
        "consumer_view_mount_id",
        "consumer_view_st_dev",
        "consumer_view_major",
        "consumer_view_minor",
    )
    if any(type(receipt[key]) is not int or receipt[key] < 0 for key in integer_fields):
        raise ValueError("mount observation has the wrong exact type")
    for key in ("runtime_lock_sha256", "consumer_view_manifest_sha256"):
        _require_lower_digest(receipt[key], 64)
    _validate_oci_digest(receipt["declared_candidate_image"])
    _require_nonempty(receipt["decision_id"])
    _require_nonempty(receipt["scope"])
    _require_nonempty(receipt["enforcement_mechanism"])
    _require_nonempty(receipt["issuer_assertion"])


def validate_isolation_receipt(
    source: bytes | str | os.PathLike[str],
    *,
    decision: ValidatedMachineRecord | Mapping[str, Any],
    lock_sha256: str,
    observation: MountObservation,
    now: datetime | None = None,
) -> ValidatedMachineRecord:
    """Validate strict egress and consumer-isolation evidence bindings.

    Args:
        source: Complete bytes or an absolute no-symlink receipt path.
        decision: Validated decision record or its data.
        lock_sha256: Complete checked-in lock digest.
        observation: Independently observed mount fields.
        now: Optional aware UTC test clock.
    Returns:
        The strict validated isolation receipt.
    Raises:
        MolmoActInferenceError: Syntax, hash, expiry, control, or binding fails.
    """

    record = read_strict_json(source, error_code="MOLMOACT_EGRESS_EVIDENCE")
    decision_data = (
        decision.data if isinstance(decision, ValidatedMachineRecord) else decision
    )
    try:
        if record.sha256 != decision_data["egress_isolation_receipt_sha256"]:
            raise ValueError("isolation receipt complete-file digest differs")
        _validate_isolation_data(
            record.data, decision_data, lock_sha256, observation, now
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_EGRESS_EVIDENCE", "isolation receipt is invalid"
        ) from exc
    return record


def _normalize_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if not normalized:
        raise ValueError("distribution name is empty")
    return normalized


def _record_rows(record_path: Path, relative_record_path: str) -> list[dict[str, str]]:
    raw_bytes = record_path.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError("RECORD has a BOM")
    text = raw_bytes.decode("utf-8", errors="strict")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in csv.reader(io.StringIO(text, newline=""), strict=True):
        if len(row) != 3:
            raise ValueError("RECORD row does not have three columns")
        path, digest, size = row
        _validate_record_row(path, digest, size, relative_record_path)
        if path in seen:
            raise ValueError("RECORD paths are duplicate")
        seen.add(path)
        rows.append({"path": path, "hash": digest, "size": size})
    if not rows:
        raise ValueError("RECORD is empty")
    return sorted(rows, key=lambda item: item["path"])


def _validate_record_row(
    path: str, digest: str, size: str, relative_record_path: str
) -> None:
    _validate_relative_path(path)
    is_own_record = path == relative_record_path
    if is_own_record:
        if bool(digest) != bool(size):
            raise ValueError("RECORD self-row hash and size disagree")
        if not digest:
            return
    if not digest or not size:
        raise ValueError("non-RECORD row lacks hash or size")
    if re.fullmatch(r"0|[1-9][0-9]*", size) is None:
        raise ValueError("RECORD size is not canonical decimal")
    if not digest.startswith("sha256="):
        raise ValueError("RECORD hash is not SHA-256")
    encoded = digest.removeprefix("sha256=")
    if "=" in encoded or re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded) is None:
        raise ValueError("RECORD hash is not unpadded URL-safe base64")
    if len(base64.urlsafe_b64decode(encoded + "=")) != 32:
        raise ValueError("RECORD hash has the wrong decoded size")


def gate_a_distribution_projection() -> list[dict[str, Any]]:
    """Reconstruct the running interpreter's complete strict distribution set.

    Args:
        None.
    Returns:
        PEP 503-sorted distribution and RECORD digest records.
    Raises:
        ValueError: Metadata or a RECORD file is missing or malformed.
    """

    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError("distribution metadata lacks name or version")
        normalized = _normalize_distribution_name(name)
        if normalized in names:
            raise ValueError("normalized distribution names are duplicate")
        names.add(normalized)
        files = [str(path).replace("\\", "/") for path in (distribution.files or ())]
        records = [path for path in files if path.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError("distribution does not have one RECORD path")
        relative = records[0]
        _validate_relative_path(relative)
        rows = _record_rows(Path(distribution.locate_file(relative)), relative)
        digest = hashlib.sha256(
            GATE_A_RECORD_DOMAIN + canonical_json_bytes(rows)
        ).hexdigest()
        result.append(
            {
                "name": name,
                "normalized_name": normalized,
                "version": version,
                "record_path": relative,
                "record_rows_count": len(rows),
                "record_rows_sha256": digest,
            }
        )
    return sorted(result, key=lambda item: item["normalized_name"])


def _validate_gate_a_distribution_schema(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError("Gate A distributions must be an array")
    names: list[str] = []
    for item in value:
        item = _require_keys(
            item,
            {
                "name",
                "normalized_name",
                "version",
                "record_path",
                "record_rows_count",
                "record_rows_sha256",
            },
        )
        if any(
            not isinstance(item[key], str) for key in item if key != "record_rows_count"
        ):
            raise ValueError("Gate A distribution field has the wrong type")
        if type(item["record_rows_count"]) is not int or item["record_rows_count"] < 0:
            raise ValueError("Gate A RECORD count has the wrong type")
        if item["normalized_name"] != _normalize_distribution_name(item["name"]):
            raise ValueError("Gate A normalized name differs")
        _validate_relative_path(item["record_path"])
        _require_lower_digest(item["record_rows_sha256"], 64)
        names.append(item["normalized_name"])
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("Gate A distributions are not unique sorted order")


def validate_gate_a_environment_receipt(
    path: str | os.PathLike[str],
    *,
    interpreter_path: str | os.PathLike[str] | None = None,
    distributions: Sequence[Mapping[str, Any]] | None = None,
) -> ValidatedMachineRecord:
    """Validate the deferred Gate A environment receipt exactly.

    Args:
        path: Absolute private no-symlink receipt path.
        interpreter_path: Exact interpreter used for Gate A.
        distributions: Optional independently reconstructed test projection.
    Returns:
        The strict validated Gate A envelope.
    Raises:
        MolmoActInferenceError: Receipt delivery, schema, digest, or binding fails.
    """

    record = read_strict_json(path, error_code="MOLMOACT_OFFLINE_STATE", private=True)
    try:
        envelope = _require_keys(record.data, {"payload", "payload_digest", "schema"})
        if envelope["schema"] != (
            "npa.molmoact.gate-a-environment-receipt-envelope.v1"
        ):
            raise ValueError("Gate A envelope schema differs")
        payload = _validate_gate_a_payload(envelope["payload"])
        _validate_gate_a_digest(payload, envelope["payload_digest"])
        executable = os.fspath(interpreter_path or sys.executable)
        _validate_gate_a_interpreter(payload["interpreter"], executable)
        observed = (
            list(distributions)
            if distributions is not None
            else (gate_a_distribution_projection())
        )
        if canonical_json_bytes(payload["distributions"]) != canonical_json_bytes(
            observed
        ):
            raise ValueError("Gate A distribution projection differs")
        versions = {item["normalized_name"]: item["version"] for item in observed}
        if versions.get("transformers") != "4.52.3":
            raise ValueError("Gate A does not use Transformers 4.52.3")
        required = {
            "torch",
            "safetensors",
            "tokenizers",
            "huggingface-hub",
            "numpy",
            "pillow",
        }
        if not required.issubset(versions):
            raise ValueError("Gate A runtime distributions are incomplete")
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE", "Gate A environment receipt is invalid"
        ) from exc
    return record


def validate_configured_gate_a_environment() -> ValidatedMachineRecord:
    """Fail-not-skip unless the external Gate A interpreter receipt is valid.

    Args:
        None.
    Returns:
        The strict Gate A environment receipt for the running interpreter.
    Raises:
        MolmoActInferenceError: Either required environment path or binding fails.
    """

    interpreter = os.environ.get("NPA_MOLMOACT_CONTRACT_PYTHON")
    receipt = os.environ.get("NPA_MOLMOACT_GATE_A_RECEIPT")
    if not interpreter or not receipt:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE",
            "Gate A interpreter and receipt are external prerequisites",
        )
    if interpreter != sys.executable:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE",
            "Gate A must run under the declared interpreter directly",
        )
    return validate_gate_a_environment_receipt(
        receipt,
        interpreter_path=interpreter,
    )


def _validate_gate_a_payload(value: Any) -> dict[str, Any]:
    payload = _require_keys(
        value,
        {
            "cpu_only_assertion",
            "created_at",
            "distributions",
            "interpreter",
            "network_disabled_assertion",
        },
    )
    if (
        type(payload["cpu_only_assertion"]) is not bool
        or payload["cpu_only_assertion"] is not True
        or type(payload["network_disabled_assertion"]) is not bool
        or payload["network_disabled_assertion"] is not True
    ):
        raise ValueError("Gate A assertions are not exact true")
    _parse_timestamp(payload["created_at"])
    _validate_gate_a_distribution_schema(payload["distributions"])
    interpreter = _require_keys(
        payload["interpreter"],
        {"abi", "implementation", "path", "python_version", "sha256"},
    )
    if any(not isinstance(item, str) for item in interpreter.values()):
        raise ValueError("Gate A interpreter fields are not strings")
    _require_lower_digest(interpreter["sha256"], 64)
    return payload


def _validate_gate_a_digest(payload: Mapping[str, Any], value: Any) -> None:
    digest = _require_keys(value, {"algorithm", "domain", "value"})
    expected = hashlib.sha256(
        GATE_A_RECEIPT_DOMAIN + canonical_json_bytes(payload)
    ).hexdigest()
    if digest != {
        "algorithm": "sha256",
        "domain": "npa.molmoact.gate-a-environment-receipt.v1\x00",
        "value": expected,
    }:
        raise ValueError("Gate A payload digest differs")


def _validate_gate_a_interpreter(value: Mapping[str, Any], executable: str) -> None:
    if executable != sys.executable:
        raise ValueError("Gate A interpreter path is not the running executable")
    raw_bytes, _ = _read_absolute_regular(executable)
    if value["path"] != executable:
        raise ValueError("Gate A interpreter path differs")
    if hashlib.sha256(raw_bytes).hexdigest() != value["sha256"]:
        raise ValueError("Gate A interpreter digest differs")
    expected = {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "abi": str(sysconfig.get_config_var("SOABI") or ""),
    }
    if any(value[key] != item for key, item in expected.items()):
        raise ValueError("Gate A interpreter observation differs")


def _hash_descriptor(descriptor: int, identity_kind: str, size: int) -> str:
    if identity_kind == "git_blob_sha1":
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(b"blob " + str(size).encode("ascii") + b"\x00")
    elif identity_kind in {"lfs_sha256", "sha256"}:
        digest = hashlib.sha256()
    else:
        raise ValueError("unknown identity kind")
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _snapshot_shape(
    root: _AnchoredDirectory, expected: set[str], category: str
) -> None:
    details = os.fstat(root.descriptor)
    if details.st_mode & 0o222:
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "snapshot root has write bits"
        )
    names = os.listdir(root.descriptor)
    if category == "qwen" and any(
        name.lower().endswith(_QWEN_WEIGHT_SUFFIXES) for name in names
    ):
        raise MolmoActInferenceError(
            "MOLMOACT_QWEN_WEIGHT_PRESENT", "Qwen closure contains a denied weight file"
        )
    if set(names) != expected:
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "snapshot exact file set differs"
        )
    if len({name.casefold() for name in names}) != len(names):
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "snapshot names case-collide"
        )


def _verify_snapshot_file(
    root_descriptor: int,
    item: Mapping[str, Any],
    category: str,
) -> tuple[int, dict[str, Any]]:
    try:
        descriptor = os.open(
            item["path"],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root_descriptor,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_mode & 0o222:
            raise MolmoActInferenceError(
                "MOLMOACT_SNAPSHOT_SHAPE", "snapshot entry is not read-only regular"
            )
        if details.st_size != item["size"]:
            raise MolmoActInferenceError(
                "MOLMOACT_SNAPSHOT_SIZE", "snapshot entry size differs"
            )
        actual = _hash_descriptor(descriptor, item["identity_kind"], details.st_size)
        if actual != item["identity"]:
            code = _identity_error_code(item["identity_kind"], category)
            raise MolmoActInferenceError(code, "snapshot entry identity differs")
        record = dict(item)
        record["verified_identity"] = actual
        return descriptor, record
    except MolmoActInferenceError:
        if "descriptor" in locals():
            os.close(descriptor)
        raise
    except (OSError, TypeError, ValueError) as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "snapshot entry cannot be opened safely"
        ) from exc


def _identity_error_code(identity_kind: str, category: str) -> str:
    if category == "input":
        return "MOLMOACT_INPUT_IDENTITY"
    if identity_kind == "git_blob_sha1":
        return "MOLMOACT_GIT_BLOB_IDENTITY"
    return "MOLMOACT_LFS_IDENTITY"


def verify_snapshot(
    root_path: str | os.PathLike[str],
    snapshot: Mapping[str, Any],
    *,
    category: str,
) -> VerifiedSnapshot:
    """Verify an exact read-only model, Qwen, or input snapshot.

    Args:
        root_path: Absolute non-symlink snapshot directory.
        snapshot: Validated lock snapshot object.
        category: ``model``, ``qwen``, or ``input``.
    Returns:
        Verified records and retained no-follow file descriptors.
    Raises:
        MolmoActInferenceError: Shape, size, identity, or Qwen denial fails.
    """

    if category not in {"model", "qwen", "input"}:
        raise ValueError("unknown snapshot category")
    try:
        root = _anchor_directory(root_path)
    except (OSError, TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "snapshot root cannot be anchored"
        ) from exc
    descriptors: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    try:
        expected = {item["path"] for item in snapshot["files"]}
        _snapshot_shape(root, expected, category)
        for item in snapshot["files"]:
            descriptor, record = _verify_snapshot_file(root.descriptor, item, category)
            descriptors[item["path"]] = descriptor
            records.append(record)
        return VerifiedSnapshot(root.path, records, descriptors)
    except MolmoActInferenceError:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "snapshot verification failed"
        ) from exc
    except BaseException:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise
    finally:
        os.close(root.descriptor)


def _open_relative_directory(root_descriptor: int, path: str) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        if path == ".":
            return descriptor
        for component in path.split("/"):
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _manifest_children(layout: Mapping[str, Any]) -> dict[str, set[str]]:
    children = {path: set() for path in layout["directories"]}
    for directory in layout["directories"]:
        if directory == ".":
            continue
        parent, name = (
            directory.rsplit("/", 1) if "/" in directory else (".", directory)
        )
        children[parent].add(name)
    for item in layout["files"]:
        parent, name = item["path"].rsplit("/", 1)
        children[parent].add(name)
    return children


def _verify_consumer_entries(
    root: _AnchoredDirectory, layout: Mapping[str, Any]
) -> dict[str, int]:
    children = _manifest_children(layout)
    for directory, expected in children.items():
        descriptor = _open_relative_directory(root.descriptor, directory)
        try:
            details = os.fstat(descriptor)
            names = os.listdir(descriptor)
            if directory == "qwen_tokenizer" and any(
                name.lower().endswith(_QWEN_WEIGHT_SUFFIXES) for name in names
            ):
                raise MolmoActInferenceError(
                    "MOLMOACT_QWEN_WEIGHT_PRESENT",
                    "consumer Qwen closure contains a denied weight file",
                )
            if details.st_mode & 0o222 or set(names) != expected:
                raise ValueError("consumer directory exact set or mode differs")
        finally:
            os.close(descriptor)
    descriptors: dict[str, int] = {}
    try:
        for item in layout["files"]:
            parent, name = item["path"].rsplit("/", 1)
            directory = _open_relative_directory(root.descriptor, parent)
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory,
                )
            finally:
                os.close(directory)
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode) or details.st_mode & 0o222:
                os.close(descriptor)
                raise ValueError("consumer file is not read-only regular")
            if details.st_size != item["size"]:
                os.close(descriptor)
                raise MolmoActInferenceError(
                    "MOLMOACT_SNAPSHOT_SIZE", "consumer file size differs"
                )
            actual = _hash_descriptor(
                descriptor, item["identity_kind"], details.st_size
            )
            if actual != item["identity"]:
                os.close(descriptor)
                category = "input" if item["path"].startswith("inputs/") else "model"
                code = _identity_error_code(item["identity_kind"], category)
                if item["identity_kind"] == "sha256":
                    code = "MOLMOACT_SNAPSHOT_SHAPE"
                raise MolmoActInferenceError(code, "consumer file identity differs")
            descriptors[item["path"]] = descriptor
        _compare_cache_copies(descriptors, layout["files"])
        return descriptors
    except BaseException:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise


def _compare_cache_copies(
    descriptors: Mapping[str, int], files: Sequence[Mapping[str, Any]]
) -> None:
    for item in files:
        if item["kind"] != "cache_copy":
            continue
        source = descriptors[item["source_path"]]
        cached = descriptors[item["path"]]
        os.lseek(source, 0, os.SEEK_SET)
        os.lseek(cached, 0, os.SEEK_SET)
        while True:
            source_chunk = os.read(source, 1024 * 1024)
            cached_chunk = os.read(cached, 1024 * 1024)
            if source_chunk != cached_chunk:
                raise ValueError("dynamic cache copy differs from its source")
            if not source_chunk:
                break
        os.lseek(source, 0, os.SEEK_SET)
        os.lseek(cached, 0, os.SEEK_SET)


def _decode_mount_field(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    decoded = re.sub(r"\\([0-7]{3})", replace, value)
    if "\\" in decoded:
        raise ValueError("mountinfo field has an invalid escape")
    return decoded


def observe_consumer_mount(
    root_path: str,
    root_descriptor: int,
    *,
    mountinfo_text: str | None = None,
    statvfs_result: Any | None = None,
) -> MountObservation:
    """Observe the exact consumer-view mount ID, device, and read-only state.

    Args:
        root_path: Canonical no-symlink consumer-view path.
        root_descriptor: Retained descriptor for that path.
        mountinfo_text: Optional injected ``/proc/self/mountinfo`` text.
        statvfs_result: Optional injected ``os.statvfs`` result.
    Returns:
        One exact mount observation.
    Raises:
        MolmoActInferenceError: Mount identity or read-only state is invalid.
    """

    try:
        text = (
            mountinfo_text
            if mountinfo_text is not None
            else Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        )
        matches = _matching_mount_entries(text, root_path)
        if len(matches) != 1:
            raise ValueError("consumer view does not have one mountinfo entry")
        fields, separator = matches[0]
        mount_id = int(fields[0], 10)
        major_text, minor_text = fields[2].split(":", 1)
        major = int(major_text, 10)
        minor = int(minor_text, 10)
        details = os.fstat(root_descriptor)
        if (major, minor) != (os.major(details.st_dev), os.minor(details.st_dev)):
            raise ValueError("mountinfo device differs from stat")
        mount_options = tuple(fields[5].split(","))
        superblock_options = tuple(fields[separator + 3].split(","))
        statvfs_value = statvfs_result or os.fstatvfs(root_descriptor)
        read_only = bool(statvfs_value.f_flag & os.ST_RDONLY)
        if "ro" not in mount_options or "ro" not in superblock_options or not read_only:
            raise ValueError("consumer view is not observed read-only")
        return MountObservation(
            mount_id,
            details.st_dev,
            major,
            minor,
            mount_options,
            superblock_options,
            read_only,
        )
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_EGRESS_EVIDENCE", "consumer mount observation is invalid"
        ) from exc


def _matching_mount_entries(text: str, root_path: str) -> list[tuple[list[str], int]]:
    matches: list[tuple[list[str], int]] = []
    for line in text.splitlines():
        fields = line.split(" ")
        if "-" not in fields or len(fields) < 10:
            raise ValueError("mountinfo line is malformed")
        separator = fields.index("-")
        if separator < 6 or len(fields) <= separator + 3:
            raise ValueError("mountinfo line is incomplete")
        mountpoint = _decode_mount_field(fields[4])
        if os.path.normpath(mountpoint) == root_path:
            matches.append((fields, separator))
    return matches


def verify_consumer_view(
    root_path: str | os.PathLike[str],
    lock: Mapping[str, Any],
    *,
    mountinfo_text: str | None = None,
    statvfs_result: Any | None = None,
) -> MountObservation:
    """Verify the exact 37-file consumer view and its read-only mount.

    Args:
        root_path: Absolute no-symlink consumer-view mountpoint.
        lock: Exactly validated runtime lock.
        mountinfo_text: Optional injected mountinfo for Gate C.
        statvfs_result: Optional injected statvfs result for Gate C.
    Returns:
        The locally observed mount identity and options.
    Raises:
        MolmoActInferenceError: Shape, bytes, copies, or mount controls fail.
    """

    try:
        root = _anchor_directory(root_path)
    except (OSError, TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "consumer view cannot be anchored"
        ) from exc
    descriptors: dict[str, int] = {}
    try:
        if os.fstat(root.descriptor).st_mode & 0o222:
            raise ValueError("consumer root has write bits")
        descriptors = _verify_consumer_entries(root, lock["consumer_view_layout"])
        return observe_consumer_mount(
            root.path,
            root.descriptor,
            mountinfo_text=mountinfo_text,
            statvfs_result=statvfs_result,
        )
    except MolmoActInferenceError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_SNAPSHOT_SHAPE", "consumer view manifest verification failed"
        ) from exc
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        os.close(root.descriptor)


def _check_heavy_preimports() -> None:
    present = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _HEAVY_PREFIXES
        )
    )
    if present:
        raise MolmoActInferenceError(
            "MOLMOACT_HEAVY_PREIMPORTED", "a forbidden heavy module is already loaded"
        )
    try:
        kernels_spec = importlib.util.find_spec("kernels")
    except (ImportError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_HUB_KERNELS", "the kernels distribution state is ambiguous"
        ) from exc
    if kernels_spec is not None:
        raise MolmoActInferenceError(
            "MOLMOACT_HUB_KERNELS", "the forbidden kernels distribution is installed"
        )


def _private_parent(path: str | os.PathLike[str], code: str) -> _AnchoredDirectory:
    anchored: _AnchoredDirectory | None = None
    try:
        anchored = _anchor_directory(path)
        details = os.fstat(anchored.descriptor)
        if details.st_uid != os.geteuid() or details.st_mode & 0o022:
            raise ValueError("directory ownership or mode is unsafe")
        return anchored
    except (OSError, TypeError, ValueError) as exc:
        if anchored is not None:
            os.close(anchored.descriptor)
        raise MolmoActInferenceError(code, "private directory is unsafe") from exc


def _create_scratch(
    parent_path: str | os.PathLike[str],
) -> tuple[str, _AnchoredDirectory]:
    parent = _private_parent(parent_path, "MOLMOACT_OFFLINE_STATE")
    for _ in range(32):
        name = f"molmoact-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent.descriptor)
            scratch_path = f"{parent.path}/{name}"
            current = _anchor_directory(parent.path)
            try:
                if (current.device, current.inode) != (parent.device, parent.inode):
                    os.rmdir(name, dir_fd=parent.descriptor)
                    os.fsync(parent.descriptor)
                    os.close(parent.descriptor)
                    raise MolmoActInferenceError(
                        "MOLMOACT_OFFLINE_STATE", "scratch parent changed identity"
                    )
            finally:
                os.close(current.descriptor)
            return scratch_path, parent
        except FileExistsError:
            continue
    os.close(parent.descriptor)
    raise MolmoActInferenceError(
        "MOLMOACT_OFFLINE_STATE", "cannot allocate unique scratch directory"
    )


def _configure_offline_environment(scratch: str, consumer_view: str) -> dict[str, str]:
    values = {
        "HF_HOME": f"{scratch}/hf-home",
        "HF_HUB_CACHE": f"{scratch}/hf-hub-cache",
        "HF_MODULES_CACHE": f"{consumer_view}/hf_modules",
        "TRANSFORMERS_CACHE": f"{scratch}/transformers-cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        for key, value in values.items():
            os.environ[key] = value
        sys.dont_write_bytecode = True
        if any(os.environ.get(key) != value for key, value in values.items()):
            raise ValueError("offline environment readback differs")
        if sys.dont_write_bytecode is not True:
            raise ValueError("bytecode suppression is inactive")
    except (OSError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE", "offline environment setup failed"
        ) from exc
    return values


def _network_denied(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise MolmoActInferenceError(
        "MOLMOACT_NETWORK_ATTEMPT", "Python network operation was attempted"
    )


@contextmanager
def network_tripwire() -> Iterator[None]:
    """Block guarded stdlib lookup and TCP-connect entry points.

    This catches accidental network access by the local-only model client. It
    is not a process-wide egress sandbox: datagram sends, inherited descriptors,
    subprocesses, and unlisted resolver functions remain outside this guard.

    Args:
        None.
    Returns:
        A context manager that restores the original stdlib functions.
    Raises:
        MolmoActInferenceError: A guarded lookup or TCP connection is attempted.
    """

    module_names = (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "create_connection",
    )
    module_originals = {name: getattr(socket, name) for name in module_names}
    socket_originals = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
    }
    try:
        for name in module_names:
            setattr(socket, name, _network_denied)
        socket.socket.connect = _network_denied  # type: ignore[method-assign]
        socket.socket.connect_ex = _network_denied  # type: ignore[method-assign]
        yield
    finally:
        for name, value in module_originals.items():
            setattr(socket, name, value)
        socket.socket.connect = socket_originals["connect"]  # type: ignore[method-assign]
        socket.socket.connect_ex = socket_originals["connect_ex"]  # type: ignore[method-assign]


def _openat2_number() -> int:
    if platform.system() != "Linux":
        raise OSError(errno.ENOSYS, "openat2 requires Linux")
    if platform.machine() not in {
        "x86_64",
        "amd64",
        "aarch64",
        "arm64",
        "riscv64",
    }:
        raise OSError(errno.ENOSYS, "unknown openat2 syscall number")
    return 437


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


def _openat2_from_root(root_descriptor: int, absolute_path: str) -> int:
    value, parts = _absolute_parts(absolute_path)
    del value
    relative = "/".join(parts).encode("utf-8")
    how = _OpenHow(
        flags=os.O_RDONLY | os.O_CLOEXEC,
        mode=0,
        resolve=0x08 | 0x04 | 0x02,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        _openat2_number(),
        root_descriptor,
        ctypes.c_char_p(relative),
        ctypes.byref(how),
        ctypes.sizeof(how),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), absolute_path)
    return int(result)


def _open_root_path_descriptor() -> int:
    if not hasattr(os, "O_PATH"):
        raise OSError(errno.ENOSYS, "O_PATH is unavailable")
    return os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)


def _observe_path_openat2(path: str) -> None:
    root_descriptor = _open_root_path_descriptor()
    try:
        descriptor = _openat2_from_root(root_descriptor, path)
        os.close(descriptor)
    finally:
        os.close(root_descriptor)


def _balanced_bracket_spans(text: str) -> list[str]:
    spans: list[str] = []
    depth = 0
    start = -1
    for index, character in enumerate(text):
        if character == "[":
            if depth == 0:
                start = index
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unmatched closing bracket")
            if depth == 0:
                spans.append(text[start : index + 1])
    if depth:
        raise ValueError("unmatched opening bracket")
    return spans


def _depth_grammar(text: str) -> tuple[str, list[int]]:
    if text.count("<DEPTH_START>") != 1 or text.count("<DEPTH_END>") != 1:
        raise MolmoActInferenceError(
            "MOLMOACT_DEPTH_GRAMMAR", "depth delimiters are not unique"
        )
    start = text.index("<DEPTH_START>")
    end_start = text.find("<DEPTH_END>", start + len("<DEPTH_START>"))
    if end_start < 0:
        raise MolmoActInferenceError(
            "MOLMOACT_DEPTH_GRAMMAR", "depth delimiters are reversed"
        )
    end = end_start + len("<DEPTH_END>")
    raw = text[start:end]
    body = raw.removeprefix("<DEPTH_START>").removesuffix("<DEPTH_END>")
    token_texts = re.findall(r"<DEPTH_([1-9][0-9]*)>", body)
    rebuilt = "".join(f"<DEPTH_{value}>" for value in token_texts)
    if rebuilt != body or len(token_texts) != 100:
        raise MolmoActInferenceError(
            "MOLMOACT_DEPTH_GRAMMAR", "depth span is not exactly 100 tokens"
        )
    values = [int(value, 10) for value in token_texts]
    if any(value < 1 or value > 128 for value in values):
        raise MolmoActInferenceError(
            "MOLMOACT_DEPTH_GRAMMAR", "depth token is outside [1, 128]"
        )
    if text.count("<DEPTH_") != 102:
        raise MolmoActInferenceError(
            "MOLMOACT_DEPTH_GRAMMAR", "additional depth-like tokens are present"
        )
    return raw, values


def _trace_grammar(spans: Sequence[str]) -> tuple[str, list[list[int]]]:
    candidates = [
        span for span in spans if re.match(r"\[\s*\[", span.lstrip()) is not None
    ]
    if len(candidates) != 1:
        raise MolmoActInferenceError(
            "MOLMOACT_TRACE_GRAMMAR", "numeric trace list is not unique"
        )
    raw = candidates[0]
    try:
        value = _strict_json_value(raw.encode("utf-8"))
    except (TypeError, UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_TRACE_GRAMMAR", "trace is not strict JSON integer grammar"
        ) from exc
    if not isinstance(value, list) or not 1 <= len(value) <= 256:
        raise MolmoActInferenceError(
            "MOLMOACT_TRACE_GRAMMAR", "trace point count is outside [1, 256]"
        )
    for point in value:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(type(coordinate) is not int for coordinate in point)
            or any(coordinate < 0 or coordinate >= 256 for coordinate in point)
        ):
            raise MolmoActInferenceError(
                "MOLMOACT_TRACE_GRAMMAR", "trace point shape, type, or range is invalid"
            )
    return raw, value


def _action_token_grammar(
    spans: Sequence[str],
    trace_raw: str,
    tokenizer: Any,
    model: Any,
) -> tuple[str, list[str], list[int]]:
    candidates = [
        span
        for span in spans
        if span != trace_raw and re.match(r"\[\s*\[", span.lstrip()) is None
    ]
    if len(candidates) != 1:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR", "action token list is not unique"
        )
    raw = candidates[0]
    inner = raw[1:-1]
    tokens = [_normalize_action_token(token) for token in inner.split(",")]
    numeric = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
    if (
        len(tokens) != 7
        or any(not token for token in tokens)
        or any("[" in token or "]" in token for token in tokens)
        or any(
            numeric.fullmatch(token) or token in {"NaN", "Infinity", "-Infinity"}
            for token in tokens
        )
    ):
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR",
            "action list is not seven raw nonnumeric tokens",
        )
    token_ids = [_token_id(tokenizer, token) for token in tokens]
    try:
        vocabulary_size = tokenizer.vocab_size
        bin_count = len(model.bin_centers)
    except (AttributeError, TypeError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR", "tokenizer or action bins are unavailable"
        ) from exc
    if type(vocabulary_size) is not int or type(bin_count) is not int:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR", "tokenizer or bin size is not an integer"
        )
    for token_id in token_ids:
        index = vocabulary_size - token_id - 1
        if index < 0 or index >= bin_count:
            raise MolmoActInferenceError(
                "MOLMOACT_ACTION_TOKEN_GRAMMAR", "action token ID is outside bin domain"
            )
    return raw, tokens, token_ids


def _normalize_action_token(value: str) -> str:
    token = value.strip()
    if not token:
        return token
    if token[0] in {"'", '"'}:
        quote = token[0]
        if len(token) < 3 or token[-1] != quote:
            raise MolmoActInferenceError(
                "MOLMOACT_ACTION_TOKEN_GRAMMAR",
                "action token has mismatched quotes",
            )
        token = token[1:-1]
    if "'" in token or '"' in token:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR",
            "action token contains an ambiguous quote",
        )
    return token


def _token_id(tokenizer: Any, token: str) -> int:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        unknown = getattr(tokenizer, "unk_token_id", None)
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR", "action token lookup failed"
        ) from exc
    if type(token_id) is not int or token_id == unknown:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR", "action token is missing or unknown"
        )
    return token_id


def _plain_sequence(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_plain_sequence(item) for item in value]
    return value


def _strict_structure_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_structure_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _sole_parser_value(value: Any, expected: Any, code: str, detail: str) -> None:
    try:
        parsed = _plain_sequence(value)
    except BaseException as exc:
        raise MolmoActInferenceError(code, detail) from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != 1
        or not _strict_structure_equal(parsed[0], expected)
    ):
        raise MolmoActInferenceError(code, detail)


def validate_generated_text(text: str, *, model: Any, tokenizer: Any) -> dict[str, Any]:
    """Validate generated-only depth, trace, tokens, and upstream parser results.

    Args:
        text: Exactly the decoded newly generated tokens.
        model: Loaded MolmoAct model with the frozen parser methods.
        tokenizer: Identity-injected local Qwen2 tokenizer.
    Returns:
        Raw and parsed depth, trace, token IDs, and seven action values.
    Raises:
        MolmoActInferenceError: Any grammar, cardinality, domain, or parser fails.
    """

    if not isinstance(text, str) or not text or not text.strip():
        raise MolmoActInferenceError(
            "MOLMOACT_GENERATION", "generated-only text is empty"
        )
    try:
        text.encode("utf-8", errors="strict")
        spans = _balanced_bracket_spans(text)
    except (UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_TOKEN_GRAMMAR", "generated text bracket grammar is invalid"
        ) from exc
    depth_raw, depth_values = _depth_grammar(text)
    trace_raw, trace = _trace_grammar(spans)
    action_raw, action_tokens, token_ids = _action_token_grammar(
        spans, trace_raw, tokenizer, model
    )
    try:
        model._qwen_tokenizer = tokenizer
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "local Qwen tokenizer injection failed"
        ) from exc
    if model._qwen_tokenizer is not tokenizer:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "local Qwen tokenizer identity differs"
        )
    _validate_upstream_depth_trace(model, text, depth_raw, trace)
    action_values = _validate_upstream_action(model, text, tokenizer)
    return {
        "depth_raw": depth_raw,
        "depth_token_values": depth_values,
        "trace_raw": trace_raw,
        "trace": trace,
        "action_tokens_raw": action_raw,
        "action_tokens": action_tokens,
        "action_token_ids": token_ids,
        "action_values": action_values,
    }


def _validate_upstream_depth_trace(
    model: Any, text: str, depth_raw: str, trace: list[list[int]]
) -> None:
    try:
        depth = model.parse_depth(text)
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_DEPTH_GRAMMAR", "upstream depth parser failed"
        ) from exc
    _sole_parser_value(
        depth,
        depth_raw,
        "MOLMOACT_DEPTH_GRAMMAR",
        "upstream depth parser disagrees with raw grammar",
    )
    try:
        parsed_trace = model.parse_trace(text)
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_TRACE_GRAMMAR", "upstream trace parser failed"
        ) from exc
    _sole_parser_value(
        parsed_trace,
        trace,
        "MOLMOACT_TRACE_GRAMMAR",
        "upstream trace parser disagrees with raw grammar",
    )


def _validate_upstream_action(
    model: Any, text: str, tokenizer: Any
) -> list[float | int]:
    if getattr(model, "_qwen_tokenizer", None) is not tokenizer:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "local Qwen tokenizer is not identity-bound"
        )
    try:
        dimension = model.get_action_dim("molmoact")
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "upstream action dimension lookup failed"
        ) from exc
    if type(dimension) is not int or dimension != 7:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "upstream action dimension is not seven"
        )
    try:
        parsed = _plain_sequence(model.parse_action(text, unnorm_key="molmoact"))
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "upstream action parser failed"
        ) from exc
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "upstream action parser did not return one vector"
        )
    vector = parsed[0]
    if not isinstance(vector, list) or len(vector) != 7:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "upstream action vector is not length seven"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        for value in vector
    ):
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "upstream action value is not finite real"
        )
    result = [
        int(value) if isinstance(value, int) else float(value) for value in vector
    ]
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_ACTION_PARSE", "action vector is not canonical JSON"
        ) from exc
    return result


def _import_runtime_libraries() -> SimpleNamespace:
    image_module = importlib.import_module("PIL.Image")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    hub_kernels = importlib.import_module("transformers.integrations.hub_kernels")
    return SimpleNamespace(
        image=image_module,
        torch=torch,
        transformers=transformers,
        AutoProcessor=transformers.AutoProcessor,
        AutoModelForImageTextToText=transformers.AutoModelForImageTextToText,
        Qwen2Tokenizer=transformers.Qwen2Tokenizer,
        is_hub_kernels_available=hub_kernels.is_hub_kernels_available,
    )


def _validate_runtime_libraries(libraries: SimpleNamespace) -> None:
    try:
        version = libraries.transformers.__version__
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "Transformers version is unavailable"
        ) from exc
    if version != "4.52.3":
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "Transformers is not exactly 4.52.3"
        )
    try:
        available = libraries.is_hub_kernels_available()
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_HUB_KERNELS", "Hub-kernel availability check failed"
        ) from exc
    if available is not False:
        raise MolmoActInferenceError(
            "MOLMOACT_HUB_KERNELS", "Hub kernels are available"
        )
    try:
        cuda_available = libraries.torch.cuda.is_available()
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "CUDA availability check failed"
        ) from exc
    if cuda_available is not True:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "CUDA is unavailable and no fallback is allowed"
        )


def _seed_dynamic_module_path(cache_path: str) -> None:
    if cache_path in sys.path:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE", "HF_MODULES_CACHE was already in sys.path"
        )
    sys.path.insert(0, cache_path)
    if sys.path.count(cache_path) != 1:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE", "HF_MODULES_CACHE is not unique in sys.path"
        )
    try:
        _observe_path_openat2(cache_path)
    except OSError as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE", "dynamic cache path failed no-symlink recheck"
        ) from exc


def _load_runtime_objects(
    libraries: SimpleNamespace, model_path: str, qwen_path: str
) -> tuple[Any, Any, Any]:
    try:
        _observe_path_openat2(model_path)
        processor = libraries.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )
        _observe_path_openat2(model_path)
        model = libraries.AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=libraries.torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        model.eval()
        _require_model_cuda_zero(model)
        _observe_path_openat2(qwen_path)
        tokenizer = libraries.Qwen2Tokenizer.from_pretrained(
            qwen_path,
            local_files_only=True,
        )
        model._qwen_tokenizer = tokenizer
        if model._qwen_tokenizer is not tokenizer:
            raise ValueError("Qwen tokenizer identity injection failed")
        libraries.Qwen2Tokenizer.from_pretrained = _forbid_qwen_fallback
        return processor, model, tokenizer
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "local runtime object loading failed"
        ) from exc


def _forbid_qwen_fallback(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise MolmoActInferenceError(
        "MOLMOACT_ACTION_PARSE",
        "mutable Qwen tokenizer fallback is forbidden",
    )


def _require_model_cuda_zero(model: Any) -> None:
    try:
        parameters = iter(model.parameters())
        first = next(parameters)
        last = first
        for parameter in parameters:
            last = parameter
    except (AttributeError, StopIteration, TypeError) as exc:
        raise ValueError("model does not expose parameters") from exc
    if str(first.device) != "cuda:0" or str(last.device) != "cuda:0":
        raise ValueError("model endpoints are not both on cuda:0")


def _decode_images(
    image_module: Any, image_bytes: Sequence[bytes]
) -> tuple[list[Any], list[dict[str, Any]]]:
    images: list[Any] = []
    observations: list[dict[str, Any]] = []
    try:
        for payload in image_bytes:
            with image_module.open(io.BytesIO(payload)) as source:
                source.load()
                image = source.convert("RGB")
                image.load()
            images.append(image)
            observations.append(
                {"width": image.size[0], "height": image.size[1], "mode": image.mode}
            )
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "immutable input image decoding failed"
        ) from exc
    return images, observations


def _prepare_inputs(
    processor: Any,
    images: Sequence[Any],
    prompt: str,
    torch: Any,
) -> tuple[str, dict[str, Any]]:
    try:
        chat_text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(chat_text, str) or not chat_text:
            raise ValueError("chat template did not return text")
        inputs = processor(
            images=[[images[0], images[1]]],
            text=chat_text,
            padding=True,
            return_tensors="pt",
        )
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "processor preprocessing failed"
        ) from exc
    try:
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError("processor returned an empty input mapping")
        if any(torch.is_tensor(value) is not True for value in inputs.values()):
            raise ValueError("processor returned a non-tensor input")
        moved = {name: value.to("cuda:0") for name, value in inputs.items()}
        if any(str(value.device) != "cuda:0" for value in moved.values()):
            raise ValueError("processor input is not on cuda:0")
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "processor tensor validation or transfer failed"
        ) from exc
    return chat_text, moved


def _generate_once(
    model: Any, inputs: Mapping[str, Any], processor: Any, torch: Any
) -> tuple[str, int]:
    try:
        input_ids = inputs["input_ids"]
        if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
            raise ValueError("input batch is not exactly one")
        torch.cuda.synchronize("cuda:0")
        torch.cuda.reset_peak_memory_stats("cuda:0")
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    use_cache=True,
                )
        torch.cuda.synchronize("cuda:0")
        if len(generated_ids.shape) != 2 or generated_ids.shape[0] != 1:
            raise ValueError("generated batch is not exactly one")
        generated_only_ids = generated_ids[:, input_ids.shape[1] :]
        token_count = generated_only_ids.shape[1]
        if type(token_count) is not int or not 1 <= token_count <= 256:
            raise ValueError("generated token count is outside [1, 256]")
        decoded = processor.batch_decode(
            generated_only_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise ValueError("batch decode did not return one sequence")
        return decoded[0], token_count
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_GENERATION", "bounded generation failed"
        ) from exc


def _runtime_observations(libraries: SimpleNamespace) -> dict[str, Any]:
    try:
        return _collect_runtime_observations(libraries)
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "runtime observation failed"
        ) from exc


def _collect_runtime_observations(libraries: SimpleNamespace) -> dict[str, Any]:
    torch = libraries.torch
    properties = torch.cuda.get_device_properties(0)
    driver_getter = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
    driver = driver_getter() if callable(driver_getter) else None
    return {
        "python": {
            "value": platform.python_version(),
            "method": "platform.python_version",
        },
        "packages": {
            "transformers": str(libraries.transformers.__version__),
            "torch": str(torch.__version__),
            "pillow": _installed_version("pillow"),
            "numpy": _installed_version("numpy"),
            "huggingface-hub": _installed_version("huggingface-hub"),
            "method": "module attributes and importlib.metadata.version",
        },
        "cuda_runtime": {
            "value": str(getattr(torch.version, "cuda", None)),
            "method": "torch.version.cuda",
        },
        "cuda_driver": {
            "value": driver,
            "method": "torch._C._cuda_getDriverVersion when available",
        },
        "device": {
            "index": 0,
            "name": str(properties.name),
            "total_memory": int(properties.total_memory),
            "method": "torch.cuda.get_device_properties(0)",
        },
        "memory": {
            "max_allocated": int(torch.cuda.max_memory_allocated("cuda:0")),
            "max_reserved": int(torch.cuda.max_memory_reserved("cuda:0")),
            "method": "post-sync process-local CUDA peak counters after reset",
        },
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _execute_runtime_call_graph(
    lock: Mapping[str, Any],
    consumer_view: str,
    image_bytes: Sequence[bytes],
    libraries: SimpleNamespace,
) -> dict[str, Any]:
    _validate_runtime_libraries(libraries)
    cache_path = f"{consumer_view}/hf_modules"
    _seed_dynamic_module_path(cache_path)
    load_started = time.monotonic_ns()
    processor, model, tokenizer = _load_runtime_objects(
        libraries,
        f"{consumer_view}/molmoact_model",
        f"{consumer_view}/qwen_tokenizer",
    )
    load_finished = time.monotonic_ns()
    images, image_observations = _decode_images(libraries.image, image_bytes)
    preprocess_started = time.monotonic_ns()
    chat_text, inputs = _prepare_inputs(
        processor, images, lock["operation"]["prompt"], libraries.torch
    )
    preprocess_finished = time.monotonic_ns()
    generation_started = time.monotonic_ns()
    generated_text, token_count = _generate_once(
        model, inputs, processor, libraries.torch
    )
    generation_finished = time.monotonic_ns()
    parsing_started = time.monotonic_ns()
    parsed = validate_generated_text(generated_text, model=model, tokenizer=tokenizer)
    parsing_finished = time.monotonic_ns()
    return {
        "chat_text": chat_text,
        "generated_text": generated_text,
        "generated_token_count": token_count,
        "parsed": parsed,
        "images": image_observations,
        "runtime_observations": _runtime_observations(libraries),
        "timings_ns": {
            "import_and_load": load_finished - load_started,
            "preprocessing": preprocess_finished - preprocess_started,
            "generation": generation_finished - generation_started,
            "parsing": parsing_finished - parsing_started,
        },
    }


def _load_and_execute_runtime(
    lock: Mapping[str, Any],
    consumer_view: str,
    image_bytes: Sequence[bytes],
) -> dict[str, Any]:
    import_started = time.monotonic_ns()
    try:
        libraries = _import_runtime_libraries()
    except MolmoActInferenceError:
        raise
    except BaseException as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_RUNTIME_LOAD", "heavy runtime import failed"
        ) from exc
    import_elapsed = time.monotonic_ns() - import_started
    result = _execute_runtime_call_graph(lock, consumer_view, image_bytes, libraries)
    result["timings_ns"]["import_and_load"] += import_elapsed
    return result


def build_result_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the non-self-referential canonical result envelope.

    Args:
        payload: Result payload whose schema is the frozen payload schema.
    Returns:
        Envelope with the domain-separated payload digest.
    Raises:
        ValueError: The payload schema or JSON value is invalid.
    """

    if payload.get("schema") != "npa.molmoact.inference-payload.v1":
        raise ValueError("result payload schema differs")
    payload_bytes = canonical_json_bytes(payload)
    digest = hashlib.sha256(PAYLOAD_DOMAIN + payload_bytes).hexdigest()
    return {
        "payload": dict(payload),
        "payload_digest": {
            "algorithm": "sha256",
            "domain": "npa.molmoact.inference-payload.v1\x00",
            "value": digest,
        },
        "schema": "npa.molmoact.inference-result.v1",
    }


def _validate_envelope_bytes(raw_bytes: bytes) -> dict[str, Any]:
    if not raw_bytes.endswith(b"\n") or raw_bytes.endswith(b"\n\n"):
        raise ValueError("result is not one-newline terminated")
    envelope = _strict_json_object(raw_bytes[:-1])
    envelope = _require_keys(envelope, {"payload", "payload_digest", "schema"})
    if envelope["schema"] != "npa.molmoact.inference-result.v1":
        raise ValueError("result envelope schema differs")
    expected = build_result_envelope(envelope["payload"])
    if envelope != expected or canonical_json_bytes(envelope) + b"\n" != raw_bytes:
        raise ValueError("result envelope is noncanonical or has a bad digest")
    return envelope


def _validate_destination(path: str | os.PathLike[str]) -> tuple[str, str]:
    value, parts = _absolute_parts(path)
    basename = parts[-1]
    if basename in {"", ".", ".."}:
        raise ValueError("destination basename is invalid")
    basename.encode("utf-8", errors="strict")
    parent = "/" + "/".join(parts[:-1])
    return parent, basename


def _destination_absent(directory: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise MolmoActInferenceError(
        "MOLMOACT_OUTPUT_COLLISION", "destination already exists"
    )


def preflight_destination(path: str | os.PathLike[str]) -> None:
    """Require a private anchored output parent and an absent destination.

    Args:
        path: Absolute result path.
    Returns:
        None.
    Raises:
        MolmoActInferenceError: Parent safety or destination absence fails.
    """

    try:
        parent_path, name = _validate_destination(path)
    except (TypeError, UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_DESTINATION", "destination path is invalid"
        ) from exc
    parent = _private_parent(parent_path, "MOLMOACT_DESTINATION")
    try:
        _destination_absent(parent.descriptor, name)
    finally:
        os.close(parent.descriptor)


def _fault(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _write_all(descriptor: int, raw_bytes: bytes) -> None:
    offset = 0
    while offset < len(raw_bytes):
        written = os.write(descriptor, raw_bytes[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short output write")
        offset += written


def _open_relative_regular(directory: int, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory,
    )
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("publication entry is not regular")
    return descriptor


def _readback(
    directory: int,
    name: str,
    expected_bytes: bytes,
    expected_inode: tuple[int, int],
) -> tuple[dict[str, Any], str]:
    descriptor = _open_relative_regular(directory, name)
    try:
        details = os.fstat(descriptor)
        if (details.st_dev, details.st_ino) != expected_inode:
            raise ValueError("publication inode differs")
        raw_bytes = _read_descriptor(descriptor)
    finally:
        os.close(descriptor)
    if raw_bytes != expected_bytes:
        raise ValueError("publication bytes differ")
    envelope = _validate_envelope_bytes(raw_bytes)
    return envelope, hashlib.sha256(raw_bytes).hexdigest()


def _parent_identity_matches(parent: _AnchoredDirectory) -> bool:
    try:
        current = _anchor_directory(parent.path)
    except (OSError, TypeError, ValueError):
        return False
    try:
        return (current.device, current.inode) == (parent.device, parent.inode)
    finally:
        os.close(current.descriptor)


def _entry_inode(directory: int, name: str) -> tuple[int, int] | None:
    try:
        descriptor = _open_relative_regular(directory, name)
    except FileNotFoundError:
        return None
    try:
        details = os.fstat(descriptor)
        return details.st_dev, details.st_ino
    finally:
        os.close(descriptor)


def _unlink_exact(
    directory: int,
    name: str,
    expected_inode: tuple[int, int],
) -> bool:
    actual = _entry_inode(directory, name)
    if actual is None:
        return True
    if actual != expected_inode:
        return False
    os.unlink(name, dir_fd=directory)
    return _entry_inode(directory, name) is None


def _cleanup_publication(
    parent: _AnchoredDirectory,
    temp_name: str | None,
    temp_inode: tuple[int, int] | None,
    destination_name: str,
    linked: bool,
) -> bool:
    clean = True
    try:
        if linked and temp_inode is not None:
            clean = (
                _unlink_exact(parent.descriptor, destination_name, temp_inode) and clean
            )
        if temp_name is not None and temp_inode is not None:
            clean = _unlink_exact(parent.descriptor, temp_name, temp_inode) and clean
        os.fsync(parent.descriptor)
    except (OSError, ValueError):
        return False
    return clean


def _publication_failure(
    exc: BaseException,
    *,
    parent: _AnchoredDirectory,
    destination_name: str,
    temp_name: str | None,
    temp_inode: tuple[int, int] | None,
    linked: bool,
    final_hash: str | None,
) -> MolmoActInferenceError:
    clean = _cleanup_publication(
        parent, temp_name, temp_inode, destination_name, linked
    )
    if not clean:
        if not linked:
            return MolmoActInferenceError(
                "MOLMOACT_OUTPUT_DURABILITY",
                "temporary publication cleanup could not be proven",
                final_state="absent",
                temporary_state="residual_unverified",
            )
        return MolmoActInferenceError(
            "MOLMOACT_OUTPUT_DURABILITY",
            "publication cleanup could not prove destination absence",
            final_state="residual_unverified_candidate",
            destination_name=destination_name,
            destination_inode=list(temp_inode) if temp_inode else None,
            known_final_sha256=final_hash,
        )
    if (
        isinstance(exc, MolmoActInferenceError)
        and exc.code == "MOLMOACT_OUTPUT_COLLISION"
    ):
        return exc
    if isinstance(exc, MolmoActInferenceError):
        return MolmoActInferenceError(
            exc.code,
            exc.detail,
            **exc.context,
            final_state="absent",
        )
    code = (
        "MOLMOACT_OUTPUT_READBACK"
        if isinstance(exc, (ValueError, UnicodeError))
        else "MOLMOACT_OUTPUT_DURABILITY"
    )
    return MolmoActInferenceError(
        code,
        "result publication failed",
        final_state="absent",
    )


def _final_openat2_readback(
    root_descriptor: int,
    destination_path: str,
    expected_bytes: bytes,
    expected_inode: tuple[int, int],
) -> tuple[str, int, str]:
    descriptor = _openat2_from_root(root_descriptor, destination_path)
    observed_monotonic = time.monotonic_ns()
    observed_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        details = os.fstat(descriptor)
        if (details.st_dev, details.st_ino) != expected_inode:
            raise ValueError("openat2 destination inode differs")
        raw_bytes = _read_descriptor(descriptor)
    finally:
        os.close(descriptor)
    if raw_bytes != expected_bytes:
        raise ValueError("openat2 destination bytes differ")
    _validate_envelope_bytes(raw_bytes)
    return hashlib.sha256(raw_bytes).hexdigest(), observed_monotonic, observed_utc


@dataclass
class _PublicationState:
    parent: _AnchoredDirectory
    destination_path: str
    destination_name: str
    raw_bytes: bytes
    envelope: dict[str, Any]
    root_descriptor: int
    started_ns: int
    temp_name: str | None = None
    temp_inode: tuple[int, int] | None = None
    linked: bool = False
    final_hash: str | None = None


def _prepare_publication(
    destination: str | os.PathLike[str], envelope: Mapping[str, Any]
) -> _PublicationState:
    try:
        parent_path, destination_name = _validate_destination(destination)
        raw_bytes = canonical_json_bytes(envelope) + b"\n"
        validated = _validate_envelope_bytes(raw_bytes)
    except (KeyError, TypeError, UnicodeError, ValueError) as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_DESTINATION", "destination or result envelope is invalid"
        ) from exc
    parent = _private_parent(parent_path, "MOLMOACT_DESTINATION")
    try:
        root_descriptor = _open_root_path_descriptor()
    except OSError:
        os.close(parent.descriptor)
        raise MolmoActInferenceError(
            "MOLMOACT_OUTPUT_DURABILITY", "cannot retain a root path descriptor"
        )
    return _PublicationState(
        parent,
        os.fspath(destination),
        destination_name,
        raw_bytes,
        validated,
        root_descriptor,
        time.monotonic_ns(),
    )


def _create_publication_temp(
    state: _PublicationState, fault: Callable[[str], None] | None
) -> None:
    state.temp_name = f".molmoact-{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        state.temp_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=state.parent.descriptor,
    )
    details = os.fstat(descriptor)
    state.temp_inode = (details.st_dev, details.st_ino)
    try:
        _fault(fault, "before_write")
        _write_all(descriptor, state.raw_bytes)
        _fault(fault, "before_file_fsync")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fault(fault, "before_temp_readback")
    _readback(
        state.parent.descriptor,
        state.temp_name,
        state.raw_bytes,
        state.temp_inode,
    )


def _link_publication(
    state: _PublicationState, fault: Callable[[str], None] | None
) -> None:
    if not _parent_identity_matches(state.parent):
        raise MolmoActInferenceError(
            "MOLMOACT_OUTPUT_DURABILITY", "output parent changed before commit"
        )
    if state.temp_inode is None:
        raise MolmoActInferenceError(
            "MOLMOACT_OUTPUT_DURABILITY", "temporary publication identity is absent"
        )
    _fault(fault, "before_link")
    try:
        os.link(
            state.temp_name,
            state.destination_name,
            src_dir_fd=state.parent.descriptor,
            dst_dir_fd=state.parent.descriptor,
            follow_symlinks=False,
        )
    except BaseException as exc:
        try:
            observed = _entry_inode(state.parent.descriptor, state.destination_name)
        except BaseException:
            state.linked = True
            raise
        if observed == state.temp_inode:
            state.linked = True
            raise
        if observed is not None or isinstance(exc, FileExistsError):
            raise MolmoActInferenceError(
                "MOLMOACT_OUTPUT_COLLISION", "destination won a publication race"
            ) from exc
        raise
    state.linked = True
    _fault(fault, "after_link")


def _durably_finish_link(
    state: _PublicationState, fault: Callable[[str], None] | None
) -> None:
    _, state.final_hash = _readback(
        state.parent.descriptor,
        state.destination_name,
        state.raw_bytes,
        state.temp_inode,
    )
    _fault(fault, "before_temp_unlink")
    if not _unlink_exact(state.parent.descriptor, state.temp_name, state.temp_inode):
        raise OSError(errno.EIO, "temporary publication inode changed")
    state.temp_name = None
    _fault(fault, "before_directory_fsync")
    os.fsync(state.parent.descriptor)
    if _entry_inode(state.parent.descriptor, state.destination_name) != (
        state.temp_inode
    ):
        raise ValueError("destination changed after directory fsync")


def _publication_receipt(
    state: _PublicationState, fault: Callable[[str], None] | None
) -> dict[str, Any]:
    _fault(fault, "before_openat2")
    final_hash, observed_monotonic, observed_utc = _final_openat2_readback(
        state.root_descriptor,
        state.destination_path,
        state.raw_bytes,
        state.temp_inode,
    )
    return {
        "schema": "npa.molmoact.publication-receipt.v1",
        "publication_verified": True,
        "lexical_destination_verified_at_openat2": True,
        "openat2_observed_monotonic_ns": observed_monotonic,
        "openat2_observed_at": observed_utc,
        "publication_elapsed_ns": time.monotonic_ns() - state.started_ns,
        "final_file_sha256": final_hash,
        "payload_sha256": state.envelope["payload_digest"]["value"],
        "anchored_directory_device": state.parent.device,
        "anchored_directory_inode": state.parent.inode,
        "destination_device": state.temp_inode[0],
        "destination_inode": state.temp_inode[1],
    }


def publish_result(
    destination: str | os.PathLike[str],
    envelope: Mapping[str, Any],
    *,
    fault: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Publish one canonical envelope with atomic hard-link no-replace semantics.

    Args:
        destination: Absolute final result path.
        envelope: Result from :func:`build_result_envelope`.
        fault: Optional private test hook called at named durability stages.
    Returns:
        Separate publication receipt proving final openat2 observation.
    Raises:
        MolmoActInferenceError: Safety, collision, durability, or readback fails.
    """

    state = _prepare_publication(destination, envelope)
    try:
        _destination_absent(state.parent.descriptor, state.destination_name)
        _create_publication_temp(state, fault)
        _link_publication(state, fault)
        _durably_finish_link(state, fault)
        return _publication_receipt(state, fault)
    except BaseException as exc:
        error = _publication_failure(
            exc,
            parent=state.parent,
            destination_name=state.destination_name,
            temp_name=state.temp_name,
            temp_inode=state.temp_inode,
            linked=state.linked,
            final_hash=state.final_hash,
        )
        raise error from exc
    finally:
        os.close(state.root_descriptor)
        os.close(state.parent.descriptor)


def _read_verified_inputs(snapshot: VerifiedSnapshot) -> list[bytes]:
    payloads: list[bytes] = []
    for record in snapshot.records:
        descriptor = snapshot.descriptors[record["path"]]
        raw_bytes = _read_descriptor(descriptor)
        if len(raw_bytes) != record["size"]:
            raise MolmoActInferenceError(
                "MOLMOACT_INPUT_IDENTITY", "input changed during retained read"
            )
        if hashlib.sha256(raw_bytes).hexdigest() != record["identity"]:
            raise MolmoActInferenceError(
                "MOLMOACT_INPUT_IDENTITY", "input changed during retained read"
            )
        payloads.append(raw_bytes)
    return payloads


def _cleanup_scratch(scratch: str, parent: _AnchoredDirectory) -> None:
    try:
        shutil.rmtree(scratch)
        if os.path.lexists(scratch):
            raise OSError(errno.EIO, "scratch remains present")
        os.fsync(parent.descriptor)
    except OSError as exc:
        raise MolmoActInferenceError(
            "MOLMOACT_OFFLINE_STATE", "scratch cleanup failed before publication"
        ) from exc


def _verified_source_payload(
    lock: Mapping[str, Any], snapshots: Sequence[VerifiedSnapshot]
) -> list[dict[str, Any]]:
    lock_snapshots = (
        lock["model_snapshot"],
        lock["qwen_snapshot"],
        lock["input_snapshot"],
    )
    return [
        {
            "repository": snapshot_lock["repository"],
            "revision": snapshot_lock["revision"],
            "files": snapshot.records,
        }
        for snapshot_lock, snapshot in zip(lock_snapshots, snapshots, strict=True)
    ]


def _operation_result_payload(
    lock: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "instruction": lock["operation"]["instruction"],
        "prompt": lock["operation"]["prompt"],
        "prompt_sha256": _PROMPT_SHA256,
        "operation_sha256": _OPERATION_SHA256,
        "chat_template_result_sha256": hashlib.sha256(
            runtime["chat_text"].encode("utf-8")
        ).hexdigest(),
        "generation": lock["operation"]["generation"],
    }


def _input_observation_payload(
    lock: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "path": file_record["path"],
            "sha256": file_record["identity"],
            "size": file_record["size"],
            **image_record,
        }
        for file_record, image_record in zip(
            lock["input_snapshot"]["files"], runtime["images"], strict=True
        )
    ]


def _consumer_observation_payload(mount: MountObservation) -> dict[str, Any]:
    return {
        "mount_id": mount.mount_id,
        "st_dev": mount.st_dev,
        "major": mount.major,
        "minor": mount.minor,
        "mount_options": list(mount.mount_options),
        "superblock_options": list(mount.superblock_options),
        "method": "stat/fstatvfs and decoded /proc/self/mountinfo",
    }


def _non_claims() -> list[str]:
    return [
        "No image attestation or promotion is established.",
        "No legal assertion or isolation issuer is authenticated.",
        "No policy quality, safety, simulation, or physical success is established.",
        "Python socket tripwires do not prove native or subprocess isolation.",
    ]


def _result_payload(
    *,
    lock_record: ValidatedMachineRecord,
    decision: ValidatedMachineRecord,
    isolation: ValidatedMachineRecord,
    candidate_image: str,
    snapshots: Sequence[VerifiedSnapshot],
    mount: MountObservation,
    runtime: Mapping[str, Any],
    verification_ns: int,
) -> dict[str, Any]:
    lock = lock_record.data
    return {
        "schema": "npa.molmoact.inference-payload.v1",
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision": {
            "decision_id": decision.data["decision_id"],
            "complete_file_sha256": decision.sha256,
        },
        "runtime_lock": {
            "path": str(RUNTIME_LOCK_PATH),
            "sha256": lock_record.sha256,
        },
        "verified_sources": _verified_source_payload(lock, snapshots),
        "inputs": _input_observation_payload(lock, runtime),
        "operation": _operation_result_payload(lock, runtime),
        "generation": {
            "generated_token_count": runtime["generated_token_count"],
            "generated_only_text": runtime["generated_text"],
            **runtime["parsed"],
        },
        "timings_monotonic_ns": {
            "verification": verification_ns,
            **runtime["timings_ns"],
        },
        "publication_timing": {
            "value": None,
            "method": "measured in the required external publication receipt",
        },
        "runtime_observations": runtime["runtime_observations"],
        "consumer_view_observation": _consumer_observation_payload(mount),
        "declared_candidate_image": {
            "value": candidate_image,
            "claim": "unverified external launcher assertion",
        },
        "egress_isolation_receipt": {
            "complete_file_sha256": isolation.sha256,
            "claim": "unverified external issuer and isolation assertions",
        },
        "external_publication_receipt_required": True,
        "non_claims": _non_claims(),
    }


@dataclass(frozen=True)
class _InferenceContext:
    lock: ValidatedMachineRecord
    decision: ValidatedMachineRecord
    isolation: ValidatedMachineRecord
    candidate_image: str
    consumer_path: str
    mount: MountObservation
    snapshots: list[VerifiedSnapshot]
    verification_ns: int


def _preflight_inference(
    *,
    decision_path: str | os.PathLike[str],
    isolation_receipt_path: str | os.PathLike[str],
    consumer_view: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    now: datetime | None,
) -> _InferenceContext:
    started = time.monotonic_ns()
    lock = load_runtime_lock()
    decision = validate_decision(
        decision_path, lock=lock.data, lock_sha256=lock.sha256, now=now
    )
    candidate = validate_image_declaration(
        os.environ.get("NPA_TASK_IMAGE"), decision.data
    )
    consumer_path = os.fspath(consumer_view)
    mount = verify_consumer_view(consumer_path, lock.data)
    isolation = validate_isolation_receipt(
        isolation_receipt_path,
        decision=decision,
        lock_sha256=lock.sha256,
        observation=mount,
        now=now,
    )
    preflight_destination(output_path)
    snapshots = _verify_all_snapshots(consumer_path, lock.data)
    return _InferenceContext(
        lock,
        decision,
        isolation,
        candidate,
        consumer_path,
        mount,
        snapshots,
        time.monotonic_ns() - started,
    )


def _execute_in_scratch(
    context: _InferenceContext, scratch_parent: str | os.PathLike[str]
) -> dict[str, Any]:
    scratch, anchor = _create_scratch(scratch_parent)
    active_error: BaseException | None = None
    try:
        _configure_offline_environment(scratch, context.consumer_path)
        image_bytes = _read_verified_inputs(context.snapshots[2])
        with network_tripwire():
            runtime = _load_and_execute_runtime(
                context.lock.data, context.consumer_path, image_bytes
            )
            _cleanup_scratch(scratch, anchor)
            scratch = ""
            return _result_payload(
                lock_record=context.lock,
                decision=context.decision,
                isolation=context.isolation,
                candidate_image=context.candidate_image,
                snapshots=context.snapshots,
                mount=context.mount,
                runtime=runtime,
                verification_ns=context.verification_ns,
            )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        if scratch:
            try:
                _cleanup_scratch(scratch, anchor)
            except MolmoActInferenceError as cleanup_error:
                if active_error is not None:
                    raise cleanup_error from active_error
                raise
        os.close(anchor.descriptor)


def run_inference(
    *,
    decision_path: str | os.PathLike[str],
    isolation_receipt_path: str | os.PathLike[str],
    consumer_view: str | os.PathLike[str],
    scratch_parent: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the private one-shot inference only after every external gate.

    Args:
        decision_path: Strict external bounded-use decision.
        isolation_receipt_path: Strict external egress/isolation receipt.
        consumer_view: Externally prepared read-only exact consumer mount.
        scratch_parent: Private caller-owned scratch parent.
        output_path: Absent final result path in a private directory.
        now: Optional aware UTC test clock for external-record expiry.
    Returns:
        The separate durable publication receipt.
    Raises:
        MolmoActInferenceError: Any contract gate or runtime operation fails.
    """

    _check_heavy_preimports()
    context = _preflight_inference(
        decision_path=decision_path,
        isolation_receipt_path=isolation_receipt_path,
        consumer_view=consumer_view,
        output_path=output_path,
        now=now,
    )
    try:
        with network_tripwire():
            payload = _execute_in_scratch(context, scratch_parent)
            return publish_result(output_path, build_result_envelope(payload))
    finally:
        for snapshot in context.snapshots:
            snapshot.close()


def _verify_all_snapshots(
    consumer_path: str, lock: Mapping[str, Any]
) -> list[VerifiedSnapshot]:
    snapshots: list[VerifiedSnapshot] = []
    try:
        for directory, key, category in (
            ("molmoact_model", "model_snapshot", "model"),
            ("qwen_tokenizer", "qwen_snapshot", "qwen"),
            ("inputs", "input_snapshot", "input"),
        ):
            snapshots.append(
                verify_snapshot(
                    f"{consumer_path}/{directory}",
                    lock[key],
                    category=category,
                )
            )
        return snapshots
    except BaseException:
        for snapshot in snapshots:
            snapshot.close()
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Private one-shot MolmoAct offline inference contract."
    )
    parser.add_argument("--decision", required=True)
    parser.add_argument("--isolation-receipt", required=True)
    parser.add_argument("--consumer-view", required=True)
    parser.add_argument("--scratch-parent", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the private process-terminal command-line entry point.

    Args:
        argv: Optional private command arguments.
    Returns:
        Zero after a verified publication; two after a stable contract failure.
    Raises:
        SystemExit: Argument parsing fails when called by ``__main__``.
    """

    arguments = _build_parser().parse_args(argv)
    try:
        receipt = run_inference(
            decision_path=arguments.decision,
            isolation_receipt_path=arguments.isolation_receipt,
            consumer_view=arguments.consumer_view,
            scratch_parent=arguments.scratch_parent,
            output_path=arguments.output,
        )
    except MolmoActInferenceError as exc:
        print(
            canonical_json_bytes(
                {"schema": "npa.molmoact.inference-error.v1", "error": exc.as_dict()}
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
