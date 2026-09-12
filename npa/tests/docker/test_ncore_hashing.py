"""Verify NCore artifact hashing without Python 3.11's file_digest API."""

import hashlib
import importlib.util
import io
from pathlib import Path
import sys

import pytest

from npa.deploy import ncore_component_advisories
from npa.workflows import ncore_runtime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from ncore_publication import process  # noqa: E402


@pytest.fixture
def sources():
    path = ROOT / "npa/docker/workbench/ncore/base_sources.py"
    spec = importlib.util.spec_from_file_location("ncore_hashing_sources", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("component", ["runtime", "source", "publication", "advisories"])
@pytest.mark.parametrize("payload", [b"", b"artifact bytes", bytes(range(256)) * 8193],
                         ids=["empty", "small", "multiple-chunks"])
def test_file_hashes_all_bytes_without_file_digest(monkeypatch, tmp_path, sources, component, payload):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    path = tmp_path / "artifact"
    path.write_bytes(payload)
    hash_file = {
        "runtime": ncore_runtime.sha256,
        "source": sources.file_digest,
        "publication": process.file_sha,
        "advisories": ncore_component_advisories._file_hash,
    }[component]

    assert hash_file(path) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("component", ["source", "publication"])
def test_stream_hash_consumes_short_reads_from_current_position(sources, component):
    class ShortReads(io.BytesIO):
        def read(self, size=-1):
            assert size > 0
            return super().read(min(size, 7))

    payload = b"the complete remaining stream contents"
    stream = ShortReads(b"prefix" + payload)
    stream.seek(len(b"prefix"))
    hash_stream = sources._stream_digest if component == "source" else process.stream_sha

    assert hash_stream(stream) == hashlib.sha256(payload).hexdigest()
    assert stream.tell() == len(b"prefix" + payload)
    assert not stream.closed
