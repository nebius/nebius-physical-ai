"""Reuse verified scene and audio outputs by their content and render settings."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def _hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _fingerprint(inputs):
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cached(directory):
    receipt = directory / "cache-receipt.json"
    if not receipt.is_file():
        return False
    try:
        files = json.loads(receipt.read_text())["files"]
        return bool(files) and all(
            Path(name).name == name and (directory / name).is_file()
            and _hash(directory / name) == digest for name, digest in files.items()
        )
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return False


def _cache_path(root, kind, inputs):
    return root / kind / _fingerprint(inputs)


def _build_cached(root, kind, inputs, build):
    directory = _cache_path(root, kind, inputs)
    if _cached(directory):
        return directory, True
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".render-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        build(staging)
        files = {path.name: _hash(path) for path in staging.iterdir() if path.is_file()}
        if not files:
            raise ValueError(f"The {kind} build produced no files")
        directory.mkdir(exist_ok=True)
        for name in files:
            (staging / name).replace(directory / name)
        receipt = staging / "cache-receipt.json"
        receipt.write_text(json.dumps({"files": files}, indent=2) + "\n")
        receipt.replace(directory / receipt.name)
    return directory, False


@contextmanager
def _render_lock(directory):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
