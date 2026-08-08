"""Validated immutable custom assets and teleoperator contracts for LeIsaac."""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any
from urllib.parse import urlparse

BUNDLE_SCHEMA = "npa.leisaac.bundle.v1"
MAX_BUNDLE_FILES = 24
MAX_BUNDLE_BYTES = 12 * 1024 * 1024
MAX_FILE_BYTES = 6 * 1024 * 1024
MAX_BUNDLE_RESULTS = 50
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_KINDS = frozenset({"robot", "scene", "device"})
_ASSET_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".py", ".json"})
_DEVICE_SUFFIXES = frozenset({".py", ".json"})
_SAFE_IMPORTS = ("dataclasses", "isaaclab", "leisaac", "numpy", "typing")
DEVICE_SCHEMA = "npa.leisaac.so101-device.v1"
DEVICE_ACTION_ORDER = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "shoulder_pan",
    "gripper",
)


class BundleError(ValueError):
    """A bounded bundle request or immutable object is invalid."""

    def __init__(self, detail: str, *, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _safe_path(value: Any) -> str:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw
        or len(raw) > 160
        or raw.startswith("/")
        or "\\" in raw
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(not _NAME.fullmatch(part) for part in path.parts)
    ):
        raise BundleError("bundle file path is unsafe")
    return path.as_posix()


def _literal_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes, int, float, bool, type(None)))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_literal_expression(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _literal_expression(key) and _literal_expression(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _literal_expression(node.operand)
    return False


def validate_declarative_python(content: bytes) -> None:
    """Allow metadata/config declarations, never executable uploaded code."""

    try:
        source = content.decode("utf-8")
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise BundleError("Python asset contract is not valid UTF-8 Python") from exc
    if len(source) > 256 * 1024:
        raise BundleError("Python asset contract is too large")
    for index, statement in enumerate(tree.body):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.Import):
            names = [alias.name for alias in statement.names]
            if all(
                any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in _SAFE_IMPORTS
                )
                for name in names
            ):
                continue
        if isinstance(statement, ast.ImportFrom):
            module = str(statement.module or "")
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in _SAFE_IMPORTS
            ):
                continue
        if isinstance(statement, ast.Assign) and _literal_expression(statement.value):
            if all(
                isinstance(target, ast.Name) and _NAME.fullmatch(target.id)
                for target in statement.targets
            ):
                continue
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
            and _literal_expression(statement.value)
        ):
            continue
        raise BundleError(
            "uploaded Python is declarative-only; calls, functions, classes, attribute writes, and executable statements are rejected"
        )


def validate_device_descriptor(content: bytes) -> None:
    """Validate the declarative contract consumed by direct SO-101 ingress."""

    try:
        document = json.loads(content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError("device entrypoint is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "driver",
        "action_order",
        "rate_hz",
    }:
        raise BundleError("device entrypoint fields are invalid")
    if (
        document.get("schema") != DEVICE_SCHEMA
        or document.get("driver") != "custom-so101"
        or tuple(document.get("action_order") or ()) != DEVICE_ACTION_ORDER
        or isinstance(document.get("rate_hz"), bool)
        or not isinstance(document.get("rate_hz"), int)
        or not 1 <= int(document["rate_hz"]) <= 120
    ):
        raise BundleError("device entrypoint contract is invalid")


def validate_bundle(payload: Any) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "name",
        "kind",
        "entrypoint",
        "files",
    }:
        raise BundleError("bundle request fields are invalid")
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise BundleError("bundle schema is unsupported")
    name = str(payload.get("name") or "")
    kind = str(payload.get("kind") or "")
    entrypoint = _safe_path(payload.get("entrypoint"))
    raw_files = payload.get("files")
    if not _NAME.fullmatch(name) or kind not in _KINDS:
        raise BundleError("bundle name or kind is invalid")
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= MAX_BUNDLE_FILES:
        raise BundleError("bundle must contain a bounded non-empty file list")
    suffixes = _DEVICE_SUFFIXES if kind == "device" else _ASSET_SUFFIXES
    files: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    total = 0
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "content_base64",
            "sha256",
        }:
            raise BundleError("bundle file fields are invalid")
        path = _safe_path(item.get("path"))
        if path in seen or PurePosixPath(path).suffix.lower() not in suffixes:
            raise BundleError("bundle contains a duplicate or unsupported file")
        try:
            content = base64.b64decode(
                str(item.get("content_base64") or ""), validate=True
            )
        except (ValueError, binascii.Error) as exc:
            raise BundleError("bundle file is not valid base64") from exc
        checksum = hashlib.sha256(content).hexdigest()
        if (
            not content
            or len(content) > MAX_FILE_BYTES
            or not _SHA256.fullmatch(str(item.get("sha256") or ""))
            or checksum != item["sha256"]
        ):
            raise BundleError("bundle file size or checksum is invalid")
        if path.endswith(".py"):
            validate_declarative_python(content)
        if path.endswith(".json"):
            try:
                document = json.loads(content)
            except (UnicodeDecodeError, ValueError) as exc:
                raise BundleError("bundle JSON file is invalid") from exc
            if not isinstance(document, dict):
                raise BundleError("bundle JSON must be an object")
        total += len(content)
        if total > MAX_BUNDLE_BYTES:
            raise BundleError("bundle exceeds the total decoded size limit")
        seen.add(path)
        files.append((path, content))
    if entrypoint not in seen:
        raise BundleError("bundle entrypoint is not present")
    entry_suffix = PurePosixPath(entrypoint).suffix.lower()
    if kind in {"robot", "scene"} and entry_suffix not in {".usd", ".usda", ".usdc"}:
        raise BundleError("robot and scene entrypoints must be USD assets")
    if kind == "device" and entry_suffix != ".json":
        raise BundleError("device entrypoint must be a declarative JSON mapping")
    if kind == "device":
        validate_device_descriptor(dict(files)[entrypoint])
    descriptors = [
        {
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        for path, content in files
    ]
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "name": name,
        "kind": kind,
        "entrypoint": entrypoint,
        "files": descriptors,
        "bytes": total,
        "python_execution": "forbidden-declarative-provenance-only",
        "device_contract": "npa.leisaac.so101-action.v1" if kind == "device" else "",
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest, files


def validate_bundle_manifest(manifest: Any, digest: str) -> dict[str, Any]:
    """Revalidate a stored manifest before the runtime materializes any bytes."""

    if not _SHA256.fullmatch(str(digest or "")):
        raise BundleError("bundle checksum is invalid")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "name",
        "kind",
        "entrypoint",
        "files",
        "bytes",
        "python_execution",
        "device_contract",
        "bundle_sha256",
    }:
        raise BundleError("bundle manifest is malformed", status_code=502)
    name = str(manifest.get("name") or "")
    kind = str(manifest.get("kind") or "")
    entrypoint = _safe_path(manifest.get("entrypoint"))
    descriptors = manifest.get("files")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("bundle_sha256") != digest
        or not _NAME.fullmatch(name)
        or kind not in _KINDS
        or manifest.get("python_execution") != "forbidden-declarative-provenance-only"
        or manifest.get("device_contract")
        != ("npa.leisaac.so101-action.v1" if kind == "device" else "")
        or not isinstance(descriptors, list)
        or not 1 <= len(descriptors) <= MAX_BUNDLE_FILES
    ):
        raise BundleError("bundle manifest is malformed", status_code=502)
    suffixes = _DEVICE_SUFFIXES if kind == "device" else _ASSET_SUFFIXES
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for item in descriptors:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise BundleError("bundle manifest is malformed", status_code=502)
        path = _safe_path(item.get("path"))
        checksum = str(item.get("sha256") or "")
        size = item.get("bytes")
        if (
            path in seen
            or PurePosixPath(path).suffix.lower() not in suffixes
            or not _SHA256.fullmatch(checksum)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 1 <= size <= MAX_FILE_BYTES
        ):
            raise BundleError("bundle manifest is malformed", status_code=502)
        seen.add(path)
        total += size
        normalized.append({"path": path, "sha256": checksum, "bytes": size})
    if (
        entrypoint not in seen
        or total > MAX_BUNDLE_BYTES
        or manifest.get("bytes") != total
    ):
        raise BundleError("bundle manifest is malformed", status_code=502)
    entry_suffix = PurePosixPath(entrypoint).suffix.lower()
    if kind in {"robot", "scene"} and entry_suffix not in {".usd", ".usda", ".usdc"}:
        raise BundleError("bundle manifest is malformed", status_code=502)
    if kind == "device" and entry_suffix != ".json":
        raise BundleError("bundle manifest is malformed", status_code=502)
    canonical_manifest = dict(manifest)
    canonical_manifest.pop("bundle_sha256")
    canonical = json.dumps(
        canonical_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise BundleError("bundle manifest checksum is invalid", status_code=502)
    return manifest


class BundleStore:
    """Bounded S3 store rooted beneath the selected LeIsaac dataset prefix."""

    def __init__(self, client: Any, dataset_uri: str, *, allowed_buckets: list[str]):
        parsed = urlparse(str(dataset_uri or ""))
        prefix = parsed.path.strip("/")
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.netloc not in set(allowed_buckets)
            or not prefix
            or any(part in {"", ".", ".."} for part in prefix.split("/"))
        ):
            raise BundleError(
                "bundle storage is outside configured S3", status_code=403
            )
        self.client = client
        self.bucket = parsed.netloc
        self.prefix = prefix

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}/bundles/{suffix.lstrip('/')}"

    def _put_immutable(self, key: str, content: bytes) -> None:
        """Create an object once, treating an identical retry as success."""

        metadata = {"sha256": hashlib.sha256(content).hexdigest()}
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                Metadata=metadata,
                IfNoneMatch="*",
            )
            return
        except Exception as exc:
            response = getattr(exc, "response", {})
            error = response.get("Error", {}) if isinstance(response, dict) else {}
            code = str(error.get("Code") or "") if isinstance(error, dict) else ""
            detail = str(exc).lower()
            if code not in {"PreconditionFailed", "KeyAlreadyExists", "412"} and not any(
                marker in detail
                for marker in ("precondition failed", "keyalreadyexists")
            ):
                raise
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
            existing = body.read(len(content) + 1)
        except Exception as exc:
            raise BundleError(
                "immutable bundle retry could not be verified", status_code=502
            ) from exc
        if existing != content:
            raise BundleError("immutable bundle object conflicts", status_code=409)

    def publish(self, payload: Any) -> dict[str, Any]:
        manifest, files = validate_bundle(payload)
        digest = manifest["bundle_sha256"]
        root = f"objects/{digest}"
        for path, content in files:
            self._put_immutable(self._key(f"{root}/files/{path}"), content)
        body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        self._put_immutable(self._key(f"{root}/bundle.json"), body)
        index = {
            "schema": BUNDLE_SCHEMA,
            "bundle_sha256": digest,
            "name": manifest["name"],
            "kind": manifest["kind"],
            "entrypoint": manifest["entrypoint"],
            "bytes": manifest["bytes"],
            "manifest_uri": f"s3://{self.bucket}/{self._key(f'{root}/bundle.json')}",
        }
        index_body = (json.dumps(index, sort_keys=True) + "\n").encode()
        self._put_immutable(
            self._key(f"index/{manifest['kind']}/{digest}.json"), index_body
        )
        return index

    def list(self, *, kind: str = "") -> dict[str, Any]:
        if kind and kind not in _KINDS:
            raise BundleError("bundle kind is invalid")
        prefix = self._key("index/") + (f"{kind}/" if kind else "")
        response = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, MaxKeys=MAX_BUNDLE_RESULTS
        )
        bundles: list[dict[str, Any]] = []
        for item in response.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key.endswith(".json"):
                continue
            raw = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read(
                131073
            )
            if len(raw) > 131072:
                continue
            try:
                document = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if (
                isinstance(document, dict)
                and document.get("schema") == BUNDLE_SCHEMA
                and _SHA256.fullmatch(str(document.get("bundle_sha256") or ""))
            ):
                bundles.append(document)
        return {
            "bundles": bundles,
            "bounded": True,
            "truncated": bool(response.get("IsTruncated")),
        }

    def get(self, digest: str) -> dict[str, Any]:
        if not _SHA256.fullmatch(str(digest or "")):
            raise BundleError("bundle checksum is invalid")
        key = self._key(f"objects/{digest}/bundle.json")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            raw = response["Body"].read(131073)
        except Exception as exc:
            raise BundleError("bundle was not found", status_code=404) from exc
        try:
            manifest = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise BundleError("bundle manifest is malformed", status_code=502) from exc
        if len(raw) > 131072:
            raise BundleError("bundle manifest is malformed", status_code=502)
        return validate_bundle_manifest(manifest, digest)

    def materialize(self, digest: str, destination: Path) -> dict[str, Any]:
        """Download one immutable bundle into a digest-scoped runtime directory."""

        manifest = self.get(digest)
        destination = Path(destination)
        if destination.name != digest or destination.parent == destination:
            raise BundleError("bundle materialization destination is invalid")
        if destination.exists():
            for descriptor in manifest["files"]:
                target = destination / descriptor["path"]
                if (
                    not target.is_file()
                    or target.stat().st_size != descriptor["bytes"]
                    or hashlib.sha256(target.read_bytes()).hexdigest()
                    != descriptor["sha256"]
                ):
                    raise BundleError(
                        "cached bundle failed integrity validation", status_code=502
                    )
            return {
                **manifest,
                "root": str(destination),
                "entrypoint_path": str(destination / manifest["entrypoint"]),
            }
        temporary = destination.with_name(destination.name + ".part")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, mode=0o700)
        try:
            for descriptor in manifest["files"]:
                key = self._key(f"objects/{digest}/files/{descriptor['path']}")
                try:
                    response = self.client.get_object(Bucket=self.bucket, Key=key)
                    content = response["Body"].read(descriptor["bytes"] + 1)
                except Exception as exc:
                    raise BundleError(
                        "bundle file was not found", status_code=502
                    ) from exc
                metadata_checksum = str(
                    (response.get("Metadata") or {}).get("sha256") or ""
                )
                if (
                    len(content) != descriptor["bytes"]
                    or hashlib.sha256(content).hexdigest() != descriptor["sha256"]
                    or (metadata_checksum and metadata_checksum != descriptor["sha256"])
                ):
                    raise BundleError(
                        "bundle file failed integrity validation", status_code=502
                    )
                target = temporary / descriptor["path"]
                target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                target.write_bytes(content)
                target.chmod(0o600)
            if manifest["kind"] == "device":
                validate_device_descriptor(
                    (temporary / manifest["entrypoint"]).read_bytes()
                )
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {
            **manifest,
            "root": str(destination),
            "entrypoint_path": str(destination / manifest["entrypoint"]),
        }
