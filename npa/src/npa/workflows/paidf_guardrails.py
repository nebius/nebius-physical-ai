"""Exact runtime model caches for the native PAIDF EVG service."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from npa.clients.huggingface import validate_hf_access
from npa.workbench.cosmos.transfer import (
    GUARDRAIL_NLTK_READY_MARKER,
    _guardrail_nltk_data_path,
    prepare_guardrail_nltk_data,
)
from npa.workflows.paidf_upstream import (
    COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
    COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
    COSMOS_GUARDRAIL_MODEL,
    COSMOS_GUARDRAIL_REVISION,
    QWEN_GUARD_MODEL,
    QWEN_GUARD_REVISION,
)

EVG_RUNTIME_MODELS = (
    (COSMOS3_SUPER_IMAGE2VIDEO_MODEL, COSMOS3_SUPER_IMAGE2VIDEO_REVISION, ()),
    (
        COSMOS_GUARDRAIL_MODEL,
        COSMOS_GUARDRAIL_REVISION,
        ("blocklist/**", "face_blur_filter/Resnet50_Final.pth"),
    ),
    (QWEN_GUARD_MODEL, QWEN_GUARD_REVISION, ()),
)
QWEN_GUARDRAIL_SOURCE_SHA256 = (
    "117e834a4362ad82748c4146391ab1116039243cfe4cca94954f4c204024daea"
)
QWEN_GUARDRAIL_PATCHED_SHA256 = (
    "ab6855b91597e86d8ff57302c323b0df8e88a8bce4d16b3f990a6ee485dd6223"
)


class PaidfGuardrailError(RuntimeError):
    """Static, sanitized model staging failure."""


def _digest_document(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _qwen_guardrail_patch_bytes(original: bytes) -> bytes:
    """Apply only the two reviewed parser/exception edits."""
    text = original.decode("utf-8")
    old_verdict = (
        "        safe_label_match = re.search(safe_pattern, content)\n"
        "        label = safe_label_match.group(1) if safe_label_match else None\n"
    )
    new_verdict = (
        "        verdict_lines = [line.strip() for line in content.splitlines()\n"
        '                         if re.match(r"^Safety\\s*:", line.strip(), flags=re.IGNORECASE)]\n'
        "        if len(verdict_lines) != 1:\n"
        '            raise RuntimeError("Qwen3Guard returned no unique safety verdict")\n'
        "        safe_label_match = re.fullmatch(safe_pattern, verdict_lines[0])\n"
        "        if safe_label_match is None:\n"
        '            raise RuntimeError("Qwen3Guard returned a malformed safety verdict")\n'
        "        label = safe_label_match.group(1)\n"
    )
    old_failure = '            return True, "Unexpected error occurred when running Qwen3Guard guardrail."\n'
    new_failure = (
        '            raise RuntimeError("Qwen3Guard inference failed closed") from e\n'
    )
    if text.count(old_verdict) != 1 or text.count(old_failure) != 1:
        raise PaidfGuardrailError(
            "installed Qwen guardrail source lacks the reviewed adaptation anchors"
        )
    return (
        text.replace(old_verdict, new_verdict, 1)
        .replace(old_failure, new_failure, 1)
        .encode()
    )


def _patch_qwen_guardrail_source(original: bytes) -> bytes:
    """Keep real Qwen inference while rejecting missing verdicts and exceptions."""
    if hashlib.sha256(original).hexdigest() != QWEN_GUARDRAIL_SOURCE_SHA256:
        raise PaidfGuardrailError(
            "installed Qwen guardrail source differs from the reviewed image"
        )
    patched = _qwen_guardrail_patch_bytes(original)
    if hashlib.sha256(patched).hexdigest() != QWEN_GUARDRAIL_PATCHED_SHA256:
        raise PaidfGuardrailError(
            "Qwen guardrail adaptation differs from the reviewed patched source"
        )
    return patched


def qwen_guardrail_source_adaptation() -> dict[str, Any]:
    value = {
        "schema": "npa.paidf.evg-guardrail-source-adaptation.v1",
        "path": "cosmos_guardrail/cosmos_guardrail.py",
        "original_sha256": QWEN_GUARDRAIL_SOURCE_SHA256,
        "patched_sha256": QWEN_GUARDRAIL_PATCHED_SHA256,
        "verdict_protocol": "one complete Safety line: Safe, Unsafe, or Controversial",
        "controversial_policy": "allow-as-upstream",
        "invalid_verdict_or_inference_exception": "raise-before-generation",
    }
    value["patch_sha256"] = _digest_document(value)
    return value


def _prepare_qwen_guardrail_overlay(
    home: Path, source_package: Path | None = None
) -> Path:
    """Copy verified vendor code privately; never alter the installed environment."""
    source_package = source_package or Path(
        "/usr/local/lib/python3.12/dist-packages/cosmos_guardrail"
    )
    if source_package.is_symlink() or not source_package.is_dir():
        raise PaidfGuardrailError(
            "accepted image has no regular Cosmos guardrail package"
        )
    files: dict[str, bytes] = {}
    for path in source_package.rglob("*"):
        relative = path.relative_to(source_package)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise PaidfGuardrailError(
                "installed Cosmos guardrail package contains a source link"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PaidfGuardrailError(
                "installed Cosmos guardrail package contains a non-regular source"
            )
        files[relative.as_posix()] = path.read_bytes()
    if "cosmos_guardrail.py" not in files or "__init__.py" not in files:
        raise PaidfGuardrailError("installed Cosmos guardrail package is incomplete")
    files["cosmos_guardrail.py"] = _patch_qwen_guardrail_source(
        files["cosmos_guardrail.py"]
    )
    root = home / "npa-paidf-guardrail-code"
    destination = root / QWEN_GUARDRAIL_PATCHED_SHA256
    if root.is_symlink() or destination.is_symlink():
        raise PaidfGuardrailError(
            "Cosmos guardrail source overlay contains a directory redirect"
        )
    root.mkdir(mode=0o700, exist_ok=True)
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(dir=root))
        try:
            for relative, payload in files.items():
                output = staging / "cosmos_guardrail" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
                output.chmod(0o444)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    actual: dict[str, bytes] = {}
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise PaidfGuardrailError(
                "Cosmos guardrail source overlay contains a source link"
            )
        if path.is_dir():
            continue
        if not path.is_file() or not path.is_relative_to(
            destination / "cosmos_guardrail"
        ):
            raise PaidfGuardrailError(
                "Cosmos guardrail source overlay contains an unexpected file"
            )
        actual[path.relative_to(destination / "cosmos_guardrail").as_posix()] = (
            path.read_bytes()
        )
    if actual != files:
        raise PaidfGuardrailError(
            "Cosmos guardrail source overlay differs from the reviewed adaptation"
        )
    return destination


def _load_snapshot_manifest(
    repository: str, revision: str, patterns: tuple[str, ...]
) -> dict:
    """Read the official exact-revision path/hash/size map without credentials."""
    url = (
        f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
    )
    try:
        # These public model manifests need no token. Access to actual gated
        # payloads is checked separately before any snapshot download.
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed official HTTPS origin
            if response.geturl() != url:
                raise PaidfGuardrailError(
                    "exact model metadata redirected to another identity"
                )
            document = json.load(response)
    except (OSError, ValueError) as exc:
        raise PaidfGuardrailError(
            "exact model file metadata could not be verified"
        ) from exc
    if (
        not isinstance(document, dict)
        or document.get("sha") != revision
        or document.get("id") != repository
        or not isinstance(document.get("siblings"), list)
    ):
        raise PaidfGuardrailError(
            "model file metadata differs from the pinned repository revision"
        )
    result = {}
    for item in document["siblings"]:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            raise PaidfGuardrailError("model file metadata has a malformed path")
        path = item["rfilename"]
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or relative.as_posix() != path
            or ".." in relative.parts
        ):
            raise PaidfGuardrailError("model file metadata has an unsafe path")
        if patterns and not any(
            fnmatch.fnmatchcase(path, pattern) for pattern in patterns
        ):
            continue
        lfs = item.get("lfs")
        if lfs is not None and not isinstance(lfs, dict):
            raise PaidfGuardrailError("model file metadata has a malformed LFS record")
        digest = lfs.get("sha256") if lfs is not None else item.get("blobId")
        algorithm = "sha256" if lfs is not None else "git-sha1"
        size = item.get("size")
        if (
            path in result
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(
                r"[0-9a-f]{64}" if lfs is not None else r"[0-9a-f]{40}", digest
            )
            or (lfs is not None and lfs.get("size") != size)
        ):
            raise PaidfGuardrailError(
                "model file metadata has an invalid hash or byte size"
            )
        result[path] = {
            "content_hash": digest,
            "hash_algorithm": algorithm,
            "size_bytes": size,
        }
    if not result:
        raise PaidfGuardrailError("exact model file metadata selected no runtime files")
    return result


def require_evg_generation_runtime(value: Any) -> None:
    """Reject missing, changed, or unreviewed EVG model provenance at handoff."""
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "models",
        "offline",
        "guardrails_enabled",
        "guardrail_source_adaptation",
        "nltk_data",
        "contract_sha256",
    }:
        raise PaidfGuardrailError(
            "EVG producer has no complete generation runtime contract"
        )
    if (
        value["schema"] != "npa.paidf.evg-generation-runtime.v1"
        or value["offline"] is not True
        or value["guardrails_enabled"] is not True
        or value["guardrail_source_adaptation"] != qwen_guardrail_source_adaptation()
        or value["contract_sha256"]
        != _digest_document({k: v for k, v in value.items() if k != "contract_sha256"})
    ):
        raise PaidfGuardrailError(
            "EVG generation runtime contract changed or disables reviewed protections"
        )
    models = value["models"]
    if not isinstance(models, list) or len(models) != len(EVG_RUNTIME_MODELS):
        raise PaidfGuardrailError(
            "EVG generation runtime omits a required pinned model"
        )
    for item, (repository, revision, _) in zip(models, EVG_RUNTIME_MODELS, strict=True):
        if (
            not isinstance(item, dict)
            or set(item)
            != {"repository", "revision", "file_count", "size_bytes", "tree_sha256"}
            or item["repository"] != repository
            or item["revision"] != revision
            or type(item["file_count"]) is not int
            or item["file_count"] < 1
            or type(item["size_bytes"]) is not int
            or item["size_bytes"] < 1
            or not isinstance(item["tree_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["tree_sha256"])
        ):
            raise PaidfGuardrailError(
                "EVG generation runtime model identity or content inventory is invalid"
            )
    nltk_data = value["nltk_data"]
    if (
        not isinstance(nltk_data, dict)
        or set(nltk_data)
        != {"repository", "revision", "file_count", "tree_sha256", "regular_files"}
        or nltk_data["repository"] != COSMOS_GUARDRAIL_MODEL
        or nltk_data["revision"] != COSMOS_GUARDRAIL_REVISION
        or nltk_data["regular_files"] is not True
        or type(nltk_data["file_count"]) is not int
        or nltk_data["file_count"] < 1
        or not isinstance(nltk_data["tree_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", nltk_data["tree_sha256"])
    ):
        raise PaidfGuardrailError(
            "EVG generation runtime lacks verified regular NLTK data"
        )


def _snapshot_inventory(
    snapshot: Path, hub: Path, repository: str, revision: str, expected_files: dict
) -> dict:
    package = hub / ("models--" + repository.replace("/", "--"))
    expected = package / "snapshots" / revision
    if snapshot.is_symlink() or snapshot.resolve() != expected or not snapshot.is_dir():
        raise PaidfGuardrailError(
            "downloaded model snapshot has an unexpected identity"
        )
    blobs = package / "blobs"
    if blobs.is_symlink() or not blobs.is_dir():
        raise PaidfGuardrailError(
            "model blob directory is not confined to its repository cache"
        )
    files = {}
    for entry in sorted(snapshot.rglob("*")):
        if entry.is_dir():
            if entry.is_symlink():
                raise PaidfGuardrailError("model snapshot contains a directory symlink")
            continue
        try:
            target = entry.resolve(strict=True)
        except OSError as exc:
            raise PaidfGuardrailError(
                "model snapshot has an unresolved cache file"
            ) from exc
        if target.parent != blobs or not stat.S_ISREG(target.stat().st_mode):
            raise PaidfGuardrailError(
                "model cache file escapes its exact repository blob directory"
            )
        relative = entry.relative_to(snapshot).as_posix()
        expected_file = expected_files.get(relative)
        if expected_file is None:
            raise PaidfGuardrailError(
                "model snapshot contains a file outside its exact selected revision"
            )
        expected_hash = target.name
        size = target.stat().st_size
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1(usedforsecurity=False)
        git_sha1.update(f"blob {size}\0".encode())
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
                git_sha1.update(chunk)
        actual_hash = (
            sha256.hexdigest() if len(expected_hash) == 64 else git_sha1.hexdigest()
        )
        if (
            not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected_hash)
            or actual_hash != expected_hash
        ):
            raise PaidfGuardrailError(
                "downloaded model bytes do not match their Hugging Face content hash"
            )
        pinned_hash = (
            sha256.hexdigest()
            if expected_file["hash_algorithm"] == "sha256"
            else git_sha1.hexdigest()
        )
        if (
            pinned_hash != expected_file["content_hash"]
            or size != expected_file["size_bytes"]
        ):
            raise PaidfGuardrailError(
                "model snapshot path bytes differ from the exact pinned revision"
            )
        files[relative] = {
            "sha256": sha256.hexdigest(),
            "size_bytes": size,
        }
    if not files:
        raise PaidfGuardrailError("downloaded model snapshot is empty")
    if files.keys() != expected_files.keys():
        raise PaidfGuardrailError(
            "model snapshot omits an exact selected revision file"
        )
    return files


def _pin_cached_default(hub: Path, repository: str, revision: str) -> None:
    """Bind vendor loaders that request main inside this isolated service cache."""
    refs = hub / ("models--" + repository.replace("/", "--")) / "refs"
    if refs.is_symlink():
        raise PaidfGuardrailError(
            "model cache references contain an unsafe directory link"
        )
    refs.mkdir(parents=True, exist_ok=True)
    destination = refs / "main"
    if destination.is_symlink():
        raise PaidfGuardrailError("model default reference is a symlink")
    if destination.exists():
        if not destination.is_file() or destination.read_text().strip() != revision:
            raise PaidfGuardrailError(
                "model default reference differs from the approved revision"
            )
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=refs, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(revision)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_model_cache(hub: Path, repository: str, revision: str) -> None:
    """Reject existing cache redirects before the download client writes bytes."""
    package = hub / ("models--" + repository.replace("/", "--"))
    snapshot = package / "snapshots" / revision
    blobs = package / "blobs"
    locks = hub / ".locks"
    for directory in (
        package,
        blobs,
        package / "snapshots",
        snapshot,
        locks,
        locks / package.name,
    ):
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise PaidfGuardrailError(
                "EVG model cache contains an unsafe directory redirect"
            )
        directory.mkdir(exist_ok=True)
    for directory in (locks / package.name,):
        for entry in directory.iterdir():
            if entry.is_symlink() or not entry.is_file() or entry.stat().st_nlink != 1:
                raise PaidfGuardrailError(
                    "EVG download storage contains an unsafe file redirect"
                )
    for entry in snapshot.rglob("*"):
        if entry.is_symlink() and (entry.is_dir() or entry.resolve().parent != blobs):
            raise PaidfGuardrailError(
                "EVG model snapshot contains an unsafe cache redirect"
            )
    _pin_cached_default(hub, repository, revision)


def prepare_evg_generation_environment() -> tuple[dict[str, str], dict[str, Any]]:
    """Fetch exact model revisions and keep vendor guardrails enabled offline.

    The worker's real Hugging Face CLI owns network/cache protocol behavior. NPA
    verifies snapshot and blob identities, stages only NLTK's small data subtree
    as regular files, and confines default model references to this service's
    revision-bound cache. A source-hash-bound private code overlay makes the
    genuine Qwen guardrail reject malformed verdicts and inference exceptions;
    installed vendor files, model inference, and NLTK path security are preserved.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise PaidfGuardrailError(
            "HF_TOKEN is required for the exact gated EVG guardrail payload"
        )
    for repository, revision, probe in (
        (
            COSMOS_GUARDRAIL_MODEL,
            COSMOS_GUARDRAIL_REVISION,
            "face_blur_filter/Resnet50_Final.pth",
        ),
        (QWEN_GUARD_MODEL, QWEN_GUARD_REVISION, "model.safetensors"),
    ):
        access = validate_hf_access(token, repository, "model", revision, probe)
        if not access.ok:
            raise PaidfGuardrailError(
                "exact EVG guardrail payload access was not verified"
            )
    base = Path(
        os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface"
    ).resolve()
    closure_hash = _digest_document(EVG_RUNTIME_MODELS)
    home = base / "npa-paidf-evg-models" / closure_hash
    for path in (home.parent, home, home / "hub"):
        if path.is_symlink():
            raise PaidfGuardrailError(
                "EVG runtime cache contains an unsafe directory link"
            )
        path.mkdir(parents=True, exist_ok=True)
    hub = (home / "hub").resolve()
    environment = dict(os.environ)
    environment.pop("HUGGINGFACE_CO_STAGING", None)
    environment.update(
        HF_HOME=str(home),
        HF_HUB_CACHE=str(hub),
        HUGGINGFACE_HUB_CACHE=str(hub),
        HF_ENDPOINT="https://huggingface.co",
    )
    overlay = _prepare_qwen_guardrail_overlay(home)
    environment["PYTHONPATH"] = str(overlay) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    assets = []
    pinned_nltk_files = {}
    for repository, revision, patterns in EVG_RUNTIME_MODELS:
        expected_files = _load_snapshot_manifest(repository, revision, patterns)
        _prepare_model_cache(hub, repository, revision)
        command = [
            "hf",
            "download",
            repository,
            "--revision",
            revision,
            "--cache-dir",
            str(hub),
            "--quiet",
        ]
        for pattern in patterns:
            command.extend(["--include", pattern])
        try:
            completed = subprocess.run(
                command, env=environment, capture_output=True, text=True, check=False
            )
        except OSError as exc:
            raise PaidfGuardrailError(
                "the accepted generation image's Hugging Face CLI is unavailable"
            ) from exc
        if completed.returncode:
            raise PaidfGuardrailError("exact EVG model snapshot download failed")
        snapshot = (
            hub / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
        )
        files = _snapshot_inventory(snapshot, hub, repository, revision, expected_files)
        if repository == COSMOS_GUARDRAIL_MODEL:
            prefix = "blocklist/nltk_data/"
            pinned_nltk_files = {
                path.removeprefix(prefix): {
                    "sha256": value["sha256"],
                    "size": value["size_bytes"],
                }
                for path, value in files.items()
                if path.startswith(prefix)
            }
        _pin_cached_default(hub, repository, revision)
        assets.append(
            {
                "repository": repository,
                "revision": revision,
                "file_count": len(files),
                "size_bytes": sum(item["size_bytes"] for item in files.values()),
                "tree_sha256": _digest_document(files),
            }
        )
    guardrail_snapshot = (
        hub
        / ("models--" + COSMOS_GUARDRAIL_MODEL.replace("/", "--"))
        / "snapshots"
        / COSMOS_GUARDRAIL_REVISION
    )
    prepare_guardrail_nltk_data(
        hf_home=str(home),
        repository=COSMOS_GUARDRAIL_MODEL,
        revision=COSMOS_GUARDRAIL_REVISION,
        snapshot_path=guardrail_snapshot,
    )
    nltk_data = _guardrail_nltk_data_path(
        str(home),
        repository=COSMOS_GUARDRAIL_MODEL,
        revision=COSMOS_GUARDRAIL_REVISION,
    )
    nltk_manifest = json.loads((nltk_data / GUARDRAIL_NLTK_READY_MARKER).read_text())
    if not pinned_nltk_files or nltk_manifest.get("files") != pinned_nltk_files:
        raise PaidfGuardrailError(
            "regular NLTK bytes differ from the exact pinned model snapshot"
        )
    environment.update(
        NLTK_DATA=str(nltk_data),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    manifest = {
        "schema": "npa.paidf.evg-generation-runtime.v1",
        "models": assets,
        "offline": True,
        "guardrails_enabled": True,
        "guardrail_source_adaptation": qwen_guardrail_source_adaptation(),
        "nltk_data": {
            "repository": COSMOS_GUARDRAIL_MODEL,
            "revision": COSMOS_GUARDRAIL_REVISION,
            "file_count": nltk_manifest["file_count"],
            "tree_sha256": nltk_manifest["tree_sha256"],
            "regular_files": True,
        },
    }
    manifest["contract_sha256"] = _digest_document(manifest)
    require_evg_generation_runtime(manifest)
    return environment, manifest
