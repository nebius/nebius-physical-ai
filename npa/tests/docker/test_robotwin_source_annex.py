"""The public source annex must preserve its exact committed byte identity."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "robotwin_source_annex", ROOT / "npa/docker/workbench/robotwin/source_annex.py"
)
assert SPEC and SPEC.loader
ANNEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANNEX)


def test_source_archive_is_deterministic_and_rejects_changed_cached_bytes(
    tmp_path, monkeypatch
):
    source = b"official source bytes"
    lock = {"artifacts": [{"path": "pool/source.tar.gz", "url": "https://example.org/source",
                           "size": len(source), "sha256": hashlib.sha256(source).hexdigest()}]}
    (tmp_path / "corresponding-sources.json").write_text(json.dumps(lock))
    monkeypatch.setattr(ANNEX, "ROOT", tmp_path)
    monkeypatch.setattr(ANNEX, "urlopen", lambda url: io.BytesIO(source))
    first = ANNEX.build(tmp_path / "first")
    second = ANNEX.build(tmp_path / "second")
    assert first == second
    with tarfile.open(tmp_path / "first" / ANNEX.ARCHIVE_NAME) as archive:
        assert archive.extractfile("pool/source.tar.gz").read() == source
        assert json.load(archive.extractfile("source-lock.json")) == lock
    (tmp_path / "first/pool/source.tar.gz").write_bytes(b"changed cached bytes")
    with pytest.raises(ValueError, match="exact lock"):
        ANNEX.build(tmp_path / "first")


@pytest.mark.parametrize("mutation", [None, "manifest", "content", "size"])
def test_anonymous_source_verification_rejects_changed_public_bytes(
    tmp_path, monkeypatch, mutation
):
    source = b"source bundle"
    source_sha = "a" * 40
    expected = {"archive": ANNEX.ARCHIVE_NAME, "bytes": len(source),
                "sha256": hashlib.sha256(source).hexdigest(), "source_lock_sha256": "b" * 64}
    (tmp_path / "source-bundle.json").write_text(json.dumps(expected))
    manifest = {**expected, "source_revision": source_sha}
    if mutation == "manifest":
        manifest["source_revision"] = "c" * 40
    public_source = source
    if mutation == "content":
        public_source = b"SOURCE bundle"
    elif mutation == "size":
        public_source += b"extra"
    urls = []

    def anonymous_download(url):
        assert isinstance(url, str)  # No authenticated Request object.
        urls.append(url)
        return io.BytesIO(json.dumps(manifest).encode() if url.endswith(".json") else public_source)

    monkeypatch.setattr(ANNEX, "ROOT", tmp_path)
    monkeypatch.setattr(ANNEX, "urlopen", anonymous_download)
    if mutation:
        with pytest.raises(ValueError, match="differs"):
            ANNEX.verify_public(source_sha)
    else:
        ANNEX.verify_public(source_sha)
        assert len(urls) == 2
    assert all(f"/robotwin-sources-dev-{source_sha}/" in url for url in urls)
