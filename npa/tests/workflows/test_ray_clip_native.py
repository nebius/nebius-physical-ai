# Verify source identity and persisted artifact contracts for the plain Ray example.
"""CPU regression tests complement the separate real CUDA workload evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def example(monkeypatch):
    """Import isolated example modules without leaking module state between tests.

    Args:
        monkeypatch: Pytest import-path isolation fixture.
    Returns:
        The imported embedding module, yielded for one test.
    Raises:
        ImportError: The example cannot be imported.
    """
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("embed", "worker")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    module = importlib.import_module("embed")
    yield module
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.fixture
def rows(example):
    """Build explicit synthetic vectors without pretending to run CLIP inference.

    Args:
        example: Isolated embedding module fixture.
    Returns:
        Six image rows with distinct unit vectors for artifact checks.
    Raises:
        OSError: Test image generation fails.
    """
    result = example.worker.preprocess_shard(list(range(6)))["rows"]
    for index, row in enumerate(result):
        row["vector"] = [float(position == index) for position in range(512)]
    return result


@pytest.fixture
def persisted(example, rows, tmp_path):
    """Persist reversed input rows to exercise output ordering and integrity.

    Args:
        example: Isolated embedding module fixture.
        rows: Explicit synthetic image/vector rows.
        tmp_path: Pytest-owned artifact directory.
    Returns:
        The artifact receipt and its directory.
    Raises:
        ValueError: Artifact validation rejects the fixture.
        OSError: Test artifacts cannot be written.
    """
    report = example.save_results(tmp_path, list(reversed(rows)), len(rows))
    return report, tmp_path


def test_persisted_parquet_lance_and_retrieval_are_reopenable(persisted, rows):
    """Reopen vector formats and perform a real Lance query on persisted rows.

    Args:
        persisted: Completed artifact receipt and directory.
        rows: Expected synthetic vector rows.
    Returns:
        None.
    Raises:
        AssertionError: Persisted vectors, types or retrieval differ.
    """
    import lancedb
    import pyarrow
    import pyarrow.parquet

    report, directory = persisted
    table = pyarrow.parquet.read_table(directory / "embeddings.parquet")
    assert table["record_id"].to_pylist() == list(range(6))
    assert table.schema.field("vector").type == pyarrow.list_(pyarrow.float32(), 512)
    parquet_hash = hashlib.sha256((directory / "embeddings.parquet").read_bytes()).hexdigest()
    assert report["parquet_sha256"] == parquet_hash
    table = lancedb.connect(str(directory / "lance")).open_table("embeddings")
    assert table.count_rows() == 6
    assert table.search(rows[3]["vector"]).metric("cosine").limit(1).to_list()[0]["record_id"] == 3
    assert report["retrieval"] == json.loads((directory / "retrieval.json").read_text())
    assert all(result["query_id"] == result["top_ids"][0] for result in report["retrieval"])


def test_persisted_previews_decode_and_match_embedded_inputs(persisted, rows):
    """Decode RGB previews and compare their exact bytes with inference inputs.

    Args:
        persisted: Completed artifact receipt and directory.
        rows: Expected original-image and crop hashes.
    Returns:
        None.
    Raises:
        AssertionError: A preview differs from its embedded input.
    """
    from PIL import Image

    _, directory = persisted
    with Image.open(directory / "preview.png") as preview:
        preview.load()
        assert preview.mode == "RGB"
        assert preview.size == (448, 1344)
    for index, row in enumerate(rows):
        original = directory / "images" / f"{index:06d}-original.png"
        crop = directory / "images" / f"{index:06d}-crop.png"
        assert hashlib.sha256(original.read_bytes()).hexdigest() == row["input_sha256"]
        assert hashlib.sha256(crop.read_bytes()).hexdigest() == row["processed_sha256"]
        with Image.open(crop) as image:
            image.load()
            assert image.mode == "RGB"
            assert image.size == (224, 224)


def test_download_manifests_cover_every_persisted_artifact(example, persisted):
    """Verify both manifests against independently reopened artifact bytes.

    Args:
        example: Isolated embedding module fixture.
        persisted: Completed artifact receipt and directory.
    Returns:
        None.
    Raises:
        AssertionError: A file is missing or a recorded digest is incorrect.
    """
    _, directory = persisted
    example.write_hashes(directory)
    listed = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert digest == hashlib.sha256((directory / name).read_bytes()).hexdigest()
        listed[name] = digest
    actual = set()
    for path in directory.rglob("*"):
        if path.is_file() and path.name != "SHA256SUMS":
            actual.add(str(path.relative_to(directory)))
    assert set(listed) == actual
    expected_manifest = {name: digest for name, digest in listed.items() if name != "sha256.json"}
    assert json.loads((directory / "sha256.json").read_text()) == expected_manifest


@pytest.mark.parametrize("change", ["missing", "duplicate", "unexpected"])
def test_output_completeness_fails_before_writing(example, rows, tmp_path, change):
    """Reject missing, duplicate and unexpected records before writing artifacts.

    Args:
        example: Isolated embedding module fixture.
        rows: Valid synthetic rows to corrupt.
        tmp_path: Empty test output directory.
        change: Selected completeness violation.
    Returns:
        None.
    Raises:
        AssertionError: Invalid output is accepted or writes artifacts.
    """
    invalid_rows = copy.deepcopy(rows)
    if change == "missing":
        invalid_rows.pop()
    elif change == "duplicate":
        invalid_rows[-1]["record_id"] = 0
    else:
        invalid_rows[-1]["record_id"] = 42
    with pytest.raises(ValueError, match="record IDs"):
        example.save_results(tmp_path, invalid_rows, len(rows))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("vector", [[0.0] * 512, [float("nan")] * 512, [1.0] * 16])
def test_invalid_vectors_cannot_be_published(example, rows, tmp_path, vector):
    """Reject nonfinite, unnormalized or incorrectly sized output vectors.

    Args:
        example: Isolated embedding module fixture.
        rows: Valid synthetic image rows.
        tmp_path: Empty test output directory.
        vector: Malformed embedding to assign to every row.
    Returns:
        None.
    Raises:
        AssertionError: Invalid vectors are published.
    """
    for row in rows:
        row["vector"] = vector
    with pytest.raises(ValueError, match="512-dimensional"):
        example.save_results(tmp_path, rows, len(rows))
    assert list(tmp_path.iterdir()) == []


def test_previews_must_match_images_sent_to_inference(example, rows, tmp_path):
    """Refuse a preview whose crop identity differs from inference provenance.

    Args:
        example: Isolated embedding module fixture.
        rows: Synthetic rows whose first crop hash will be corrupted.
        tmp_path: Test output directory.
    Returns:
        None.
    Raises:
        AssertionError: Mismatched preview data is accepted.
    """
    rows[0]["processed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Preview bytes differ"):
        example.save_results(tmp_path, rows, len(rows))
    assert not (tmp_path / "preview.png").exists()


@pytest.mark.parametrize("address", [None, "", "auto", "local", "127.0.0.1:6380", "http://127.0.0.1:8265", ":6381"])
def test_driver_refuses_missing_or_management_ray(example, monkeypatch, address):
    """Prevent implicit cluster discovery or connection to management Ray.

    Args:
        example: Isolated embedding module fixture.
        monkeypatch: Pytest environment fixture.
        address: Invalid or missing application address.
    Returns:
        None.
    Raises:
        AssertionError: The driver accepts an unsafe address.
    """
    if address is None:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("RAY_ADDRESS", address)
    with pytest.raises(ValueError, match="application Ray Jobs"):
        example.application_gcs_address()


def test_driver_accepts_explicit_jobs_application_address(example, monkeypatch):
    """Accept the explicit GCS address supplied by application Ray Jobs.

    Args:
        example: Isolated embedding module fixture.
        monkeypatch: Pytest environment fixture.
    Returns:
        None.
    Raises:
        AssertionError: A valid application address is rejected or rewritten.
    """
    monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6381")
    assert example.application_gcs_address() == "127.0.0.1:6381"


def test_source_receipt_hashes_actual_delivered_files_and_refuses_missing_udf(example, tmp_path, monkeypatch):
    """Require real delivered source files and notice edits without changing UDF bytes.

    Args:
        example: Isolated embedding module fixture.
        tmp_path: Temporary source directory.
        monkeypatch: Pytest module-location fixture.
    Returns:
        None.
    Raises:
        AssertionError: Missing, linked or changed source is misidentified.
    """
    monkeypatch.setattr(example, "__file__", str(tmp_path / "embed.py"))
    for name in example.SOURCE_FILES[:-1]:
        (tmp_path / name).write_text(name)
    with pytest.raises(ValueError, match="canonical Workbench UDF"):
        example.source_hashes()
    canonical_source = tmp_path / example.SOURCE_FILES[-1]
    canonical_source.write_text("canonical source bytes")
    before = example.source_hashes()
    assert before[canonical_source.name] == hashlib.sha256(canonical_source.read_bytes()).hexdigest()
    (tmp_path / "worker.py").write_text("changed source")
    after = example.source_hashes()
    assert before["worker.py"] != after["worker.py"]
    assert before[canonical_source.name] == after[canonical_source.name]
    canonical_source.unlink()
    canonical_source.symlink_to(tmp_path / "worker.py")
    with pytest.raises(ValueError, match="canonical Workbench UDF"):
        example.source_hashes()


def _install_cuda_stub(monkeypatch, available: bool) -> None:
    """Keep CUDA failure-path tests hermetic and prevent accidental GPU use."""
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace())
    cuda = SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))


def test_actor_refuses_cpu_fallback(example, monkeypatch):
    """Require CUDA before any model loading or inference attempt.

    Args:
        example: Isolated embedding module fixture.
        monkeypatch: Pytest module fixture for unavailable CUDA.
    Returns:
        None.
    Raises:
        AssertionError: The actor proceeds without an actual CUDA GPU.
    """
    _install_cuda_stub(monkeypatch, available=False)
    monkeypatch.setitem(sys.modules, "npa_lancedb_bdd100k_udfs", SimpleNamespace())
    with pytest.raises(RuntimeError, match="actual CUDA GPU"):
        example.ClipModel("/unused-model")


def test_actor_hashes_the_actually_imported_udf(example, tmp_path, monkeypatch):
    """Reject a stale imported UDF even when the selected source hash is valid.

    Args:
        example: Isolated embedding module fixture.
        tmp_path: Temporary stale-source directory.
        monkeypatch: Pytest module and expected-provenance fixture.
    Returns:
        None.
    Raises:
        AssertionError: The actor accepts an imported source mismatch.
    """
    stale_source = tmp_path / "other-runtime" / "npa_lancedb_bdd100k_udfs.py"
    stale_source.parent.mkdir()
    stale_source.write_text("stale source")
    _install_cuda_stub(monkeypatch, available=True)
    monkeypatch.setitem(sys.modules, "npa_lancedb_bdd100k_udfs", SimpleNamespace(__file__=str(stale_source)))
    expected = {
        "embed.py": example.sha256(Path(example.__file__)),
        "worker.py": example.worker.source_hash(),
        "npa_lancedb_bdd100k_udfs.py": hashlib.sha256(b"selected canonical source").hexdigest(),
    }
    monkeypatch.setattr(example, "source_hashes", lambda: expected)
    with pytest.raises(ValueError, match="Imported application/UDF modules differ"):
        example.ClipModel("/unused-model")


@pytest.mark.parametrize("content", [b"", b"source-bytes" * 100000], ids=["empty", "multi-chunk"])
def test_file_hashing_supports_python310_without_file_digest(example, tmp_path, monkeypatch, content):
    """Hash empty and multi-chunk files without Python 3.11's convenience API.

    Args:
        example: Isolated embedding module fixture.
        tmp_path: Temporary input-file directory.
        monkeypatch: Pytest compatibility-surface fixture.
        content: Empty or multi-chunk bytes with an independently computed digest.
    Returns:
        None.
    Raises:
        AssertionError: Hashing depends on the newer API or returns incorrect bytes.
    """
    monkeypatch.delattr(example.hashlib, "file_digest", raising=False)
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    assert example.sha256(source) == hashlib.sha256(content).hexdigest()
