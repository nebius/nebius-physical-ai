# Responsibility: Test CLIP source identity, checkpoint safety and persisted artifact correctness without GPU claims.
"""Hermetic contracts for the real Ray CLIP application; GPU execution is live."""

from __future__ import annotations

import asyncio
import copy
import importlib
import io
from pathlib import Path
import sys
import shutil
from types import SimpleNamespace

import pytest


@pytest.fixture
def recipe(monkeypatch):
    """Load isolated example modules for hermetic application tests.

    Args:
        monkeypatch: Fixture restoring modified modules and attributes afterward.
    Returns:
        An isolated module namespace, yielded for the test lifetime.
    Raises:
        ImportError: A required example or canonical UDF module is unavailable.
    """
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("validation", "worker", "application")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    modules = {name: importlib.import_module(name) for name in names}
    udf_source = Path(__file__).parents[2] / "src/npa/workbench/lancedb/bdd100k_udfs.py"
    yield SimpleNamespace(**modules, directory=directory, udf_source=udf_source)
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.fixture
def source_bundle(recipe):
    """Supply the ordinary source files uploaded by Ray working_dir.

    Args:
        recipe: Isolated example modules and canonical source paths.
    Returns:
        A source-copy callable returning filename-to-hash manifests.
    Raises:
        None.
    """
    def copy_to(destination):
        """Copy canonical application source into a simulated worker package.

        Args:
            destination: Temporary destination for the simulated source package.
        Returns:
            A filename-to-SHA-256 manifest of the copied source.
        Raises:
            OSError: Source files cannot be copied or read.
        """
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("application.py", "worker.py", "validation.py"):
            shutil.copy2(recipe.directory / name, destination / name)
        shutil.copy2(recipe.udf_source, destination / "npa_lancedb_bdd100k_udfs.py")
        return {path.name: recipe.validation.file_hash(path) for path in destination.iterdir()}
    return copy_to


def test_crop_edit_changes_real_pixels_and_restores_model_inputs(recipe):
    """Crop edit changes real pixels and restores model inputs.

    Args:
        recipe: Isolated example modules and canonical source paths.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    from PIL import Image

    raw = recipe.worker.render_record(42)
    assert raw == recipe.worker.render_record(42)
    recipe.worker.CROP_POLICY = "left"
    left = recipe.worker.preprocess_image(raw)
    recipe.worker.CROP_POLICY = "right"
    right = recipe.worker.preprocess_image(raw)
    assert left != right
    assert Image.open(io.BytesIO(left)).size == (224, 224)
    assert Image.open(io.BytesIO(right)).size == (224, 224)
    assert recipe.worker.render_record(42) == raw
    recipe.worker.CROP_POLICY = "left"
    assert recipe.worker.preprocess_image(raw) == left


def test_application_imports_real_udf_from_jobs_bundle(recipe, source_bundle, tmp_path, monkeypatch):
    """Application imports real udf from jobs bundle.

    Args:
        recipe: Isolated example modules and canonical source paths.
        source_bundle: Fixture that copies the ordinary Jobs source files.
        tmp_path: Temporary directory owned by this test.
        monkeypatch: Fixture restoring modified modules and attributes afterward.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: A contract assertion fails.
        pytest.fail.Exception: An expected exception is not raised.
    """
    import pyarrow as pa

    source_bundle(tmp_path)
    monkeypatch.setattr(recipe.application, "__file__", "/unavailable-driver/ray-session/application.py")
    monkeypatch.setattr(recipe.worker, "__file__", str(tmp_path / "worker.py"))
    monkeypatch.setitem(sys.modules, "npa_lancedb_bdd100k_udfs", None)
    udf = recipe.application.load_workbench_udf()
    assert Path(udf.__file__).resolve() == (tmp_path / "npa_lancedb_bdd100k_udfs.py").resolve()
    assert recipe.validation.file_hash(Path(udf.__file__)) == recipe.validation.file_hash(recipe.udf_source)
    batch = pa.record_batch({"image_bytes": [recipe.worker.render_record(42)]})
    assert udf.udf_dhash(batch).type == pa.int64()
    (tmp_path / "npa_lancedb_bdd100k_udfs.py").unlink()
    with pytest.raises(RuntimeError, match="lacks a regular Workbench CLIP UDF"):
        recipe.application.load_workbench_udf()


def _import_actor_from_separate_file(recipe, source_bundle, tmp_path, monkeypatch):
    """Ensure the actual imported application differs from its packaged neighbor."""
    bundle = tmp_path / "worker-runtime-env"
    manifest = source_bundle(bundle)
    actual_path = tmp_path / "imported_application.py"
    actual_path.write_text((bundle / "application.py").read_text() + "\n# Distinguish imported source from neighboring package bytes.\n")
    specification = importlib.util.spec_from_file_location("imported_application", actual_path)
    imported = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, specification.name, imported)
    specification.loader.exec_module(imported)
    monkeypatch.setattr(recipe.worker, "__file__", str(bundle / "worker.py"))
    monkeypatch.setattr(recipe.validation, "__file__", str(bundle / "validation.py"))
    return imported, actual_path, bundle, manifest


def _install_provenance_runtime(monkeypatch):
    """Provide GPU metadata without claiming this unit test performs inference."""
    import importlib.metadata

    context = SimpleNamespace(
        get_node_id=lambda: "worker-node",
        get_accelerator_ids=lambda: {"GPU": ["0"]},
    )
    ray = SimpleNamespace(__version__="2.58.0", get_runtime_context=lambda: context)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
        get_device_name=lambda index: "test-gpu",
        get_device_capability=lambda index: (12, 0),
    )
    torch = SimpleNamespace(
        __version__="test-version", version=SimpleNamespace(cuda="test-version"), cuda=cuda,
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "torch", torch)


def _record_model_loads(imported, monkeypatch):
    """Keep the canonical module import while recording its model-load request."""
    original_loader = imported.load_workbench_udf
    loads = []

    def _record_clip_initialization(**arguments):
        """Retain the requested model parameters without loading weights in this test."""
        loads.append(arguments)

    def load_udf():
        """Record model loading while preserving the canonical UDF import.

        Args:
            None.
        Returns:
            The imported canonical UDF module.
        Raises:
            ImportError: A required example or canonical UDF module is unavailable.
        """
        module = original_loader()
        module._clip_components = _record_clip_initialization
        return module

    monkeypatch.setattr(imported, "load_workbench_udf", load_udf)
    return loads


def _model_snapshot_for_provenance(directory):
    """Create identifiable non-model bytes for a metadata-only unit test."""
    checkpoint = directory / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"clip"}')
    (checkpoint / "pytorch_model.bin").write_bytes(b"test weights; no inference in this unit test")
    return checkpoint


def test_actor_provenance_hashes_actual_imported_application_not_neighbor(recipe, source_bundle, tmp_path, monkeypatch):
    """Actor provenance hashes actual imported application not neighbor.

    Args:
        recipe: Isolated example modules and canonical source paths.
        source_bundle: Fixture that copies the ordinary Jobs source files.
        tmp_path: Temporary directory owned by this test.
        monkeypatch: Fixture restoring modified modules and attributes afterward.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    imported, actual_path, bundle, manifest = _import_actor_from_separate_file(
        recipe, source_bundle, tmp_path, monkeypatch)
    loads = _record_model_loads(imported, monkeypatch)
    _install_provenance_runtime(monkeypatch)
    checkpoint = _model_snapshot_for_provenance(tmp_path)
    actor = imported.ClipActor(str(checkpoint), "test-model-revision")
    assert loads == [{"device": "cuda:0", "precision": "float32"}]
    assert actor.info["application_sha256"] == recipe.validation.file_hash(actual_path)
    assert actor.info["application_sha256"] != manifest["application.py"]
    assert actor.info["application_module_path"] == str(actual_path)
    assert actor.info["source_sha256"] == manifest["worker.py"]
    assert actor.info["validation_sha256"] == manifest["validation.py"]
    assert actor.info["udf_sha256"] == manifest["npa_lancedb_bdd100k_udfs.py"]
    assert Path(actor.info["udf_module_path"]).parent == bundle
    assert actor.udf.CLIP_MODEL_NAME == str(checkpoint.resolve())


@pytest.mark.parametrize("ids,count", [([0, 0], 2), ([0, 2], 2), ([1], 1)])
def test_completeness_rejects_duplicate_missing_and_extra_ids(recipe, ids, count):
    """Completeness rejects duplicate missing and extra ids.

    Args:
        recipe: Isolated example modules and canonical source paths.
        ids: Record IDs containing the parametrized completeness defect.
        count: Expected complete number of records.
    Returns:
        None after the assertions pass.
    Raises:
        pytest.fail.Exception: An expected exception is not raised.
    """
    with pytest.raises(ValueError, match="record IDs"):
        recipe.validation.verify_ids(ids, count)


def test_partition_covers_uneven_batches(recipe):
    """Partition covers uneven batches.

    Args:
        recipe: Isolated example modules and canonical source paths.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    chunks = recipe.validation.partitions(131, 64)
    assert list(map(len, chunks)) == [64, 64, 3]
    record_ids = []
    for chunk in chunks:
        record_ids.extend(chunk)
    recipe.validation.verify_ids(record_ids, 131)


def test_checkpoint_replay_skips_actor_and_corruption_fails(recipe, tmp_path):
    """Checkpoint replay skips actor and corruption fails.

    Args:
        recipe: Isolated example modules and canonical source paths.
        tmp_path: Temporary directory owned by this test.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: A contract assertion fails.
        pytest.fail.Exception: An expected exception is not raised.
    """
    shard = recipe.worker.preprocess_shard([0, 1])
    identity = recipe.validation.checkpoint_identity(shard, "revision", "execution")
    data = tmp_path / "embeddings.parquet"
    data.write_bytes(b"checkpoint bytes")
    receipt = {"identity": identity, "parquet_sha256": recipe.validation.file_hash(data)}
    recipe.validation.atomic_json(tmp_path / "commit.json", receipt)

    def _reject_duplicate_inference(*arguments):
        """Fail if checkpoint replay schedules GPU work instead of reusing its output."""
        pytest.fail("replay dispatched GPU work")

    inference = SimpleNamespace(remote=_reject_duplicate_inference)
    forbidden_actor = SimpleNamespace(infer=inference)
    cached, future = recipe.application.submit_shard(forbidden_actor, shard, tmp_path, "revision", "execution")
    assert cached["checkpoint_reused"] is True and future is None
    with pytest.raises(ValueError, match="identity differs"):
        recipe.application.submit_shard(forbidden_actor, shard, tmp_path, "different-model", "execution")
    with pytest.raises(ValueError, match="identity differs"):
        recipe.application.submit_shard(forbidden_actor, shard, tmp_path, "revision", "changed-weights-or-udf")
    data.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        recipe.application.submit_shard(forbidden_actor, shard, tmp_path, "revision", "execution")


def test_output_root_requires_matching_execution_fingerprint(recipe, tmp_path):
    """Output root requires matching execution fingerprint.

    Args:
        recipe: Isolated example modules and canonical source paths.
        tmp_path: Temporary directory owned by this test.
    Returns:
        None after the assertions pass.
    Raises:
        pytest.fail.Exception: An expected exception is not raised.
    """
    recipe.validation.verify_execution(tmp_path, "first")
    recipe.validation.verify_execution(tmp_path, "first")
    with pytest.raises(ValueError, match="different execution fingerprint"):
        recipe.validation.verify_execution(tmp_path, "different")
    (tmp_path / "execution.json").unlink()
    (tmp_path / "old-shard").write_bytes(b"old")
    with pytest.raises(ValueError, match="no execution fingerprint"):
        recipe.validation.verify_execution(tmp_path, "first")


def test_model_fingerprint_ignores_download_metadata_but_covers_model_bytes(recipe, tmp_path):
    """Model fingerprint ignores download metadata but covers model bytes.

    Args:
        recipe: Isolated example modules and canonical source paths.
        tmp_path: Temporary directory owned by this test.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"model weights")
    (tmp_path / "config.json").write_text('{"model_type":"clip"}')
    metadata = tmp_path / ".cache/huggingface/download/pytorch_model.bin.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("revision\netag\n1780000000.0\n")
    first = recipe.application.model_snapshot_files(tmp_path)
    metadata.write_text("revision\netag\n1780000001.0\n")
    assert recipe.application.model_snapshot_files(tmp_path) == first
    assert {row["path"] for row in first} == {"config.json", "pytorch_model.bin"}
    weights.write_bytes(b"different model weights")
    assert recipe.application.model_snapshot_files(tmp_path) != first


def test_cleanup_attempts_all_actors_and_shutdown_after_first_kill_failure(recipe):
    """Cleanup attempts all actors and shutdown after first kill failure.

    Args:
        recipe: Isolated example modules and canonical source paths.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    calls = []
    def kill(actor, **kwargs):
        """Record actor cleanup and raise the intended first-actor failure.

        Args:
            actor: Actor selected for cleanup.
            kwargs: Unused Ray lifecycle keyword arguments.
        Returns:
            None when the simulated actor can be stopped.
        Raises:
            RuntimeError: The deliberately unavailable first actor is selected.
        """
        calls.append(actor)
        if actor == "first":
            raise RuntimeError("actor unavailable")

    def _record_shutdown():
        """Make driver cleanup visible after the attempted actor shutdowns."""
        calls.append("shutdown")

    ray = SimpleNamespace(kill=kill, shutdown=_record_shutdown)
    errors = recipe.application.cleanup_actors(ray, ["first", "second"])
    assert calls == ["first", "second", "shutdown"]
    assert errors == [{"operation": "kill actor", "actor_index": 0, "error_type": "RuntimeError"}]


def test_uncommitted_checkpoint_is_recomputed(recipe, tmp_path):
    """Uncommitted checkpoint is recomputed.

    Args:
        recipe: Isolated example modules and canonical source paths.
        tmp_path: Temporary directory owned by this test.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    (tmp_path / "embeddings.parquet").write_bytes(b"partial without commit marker")
    shard = recipe.worker.preprocess_shard([4])
    calls = []
    def _record_inference(*arguments):
        calls.append(arguments)
        return "future"

    actor = SimpleNamespace(infer=SimpleNamespace(remote=_record_inference))
    cached, future = recipe.application.submit_shard(actor, shard, tmp_path, "revision", "execution")
    assert cached is None and future == "future" and len(calls) == 1


def test_real_parquet_checkpoint_lance_aggregation_and_retrieval(recipe, tmp_path):
    """Real parquet checkpoint lance aggregation and retrieval.

    Args:
        recipe: Isolated example modules and canonical source paths.
        tmp_path: Temporary directory owned by this test.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    import numpy as np
    import pyarrow as pa

    receipts = []
    for index in range(3):
        shard = recipe.worker.preprocess_shard([index])
        vector = np.zeros(512, dtype=np.float32)
        vector[index] = 1.0
        vectors = pa.array([vector.tolist()], type=pa.list_(pa.float32(), 512))
        path = tmp_path / "shards" / f"{index:06d}"
        receipt = recipe.application.commit_shard(path, shard, vectors, {"inference_seconds": 0.1}, "model-revision", "execution")
        assert recipe.validation.read_checkpoint(path, receipt["identity"]) == receipt
        receipts.append(receipt)
    report = recipe.application.aggregate(tmp_path, receipts, 3)
    assert report["lance_rows"] == 3
    assert report["retrieval_queries"] == 3
    assert (tmp_path / "embeddings.parquet").is_file()
    assert (tmp_path / "retrieval.json").is_file()


def _baseline_model_provenance():
    """Keep every independent model/runtime field visible in the comparison fixture."""
    return {
        "model_revision": "model",
        "model_files": [{"path": "weights.bin", "sha256": "weights"}],
        "model_config_sha256": "config",
        "udf_sha256": "udf",
        "precision": "float32",
        "gpu_capability": [10, 0],
        "python": "3.12",
        "ray": "2.58.0",
        "torch": "2.7.0",
        "cuda": "12.8",
        "transformers": "4.49.0",
        "pyarrow": "19.0.0",
        "lancedb": "0.21.0",
        "source_sha256": "left",
        "application_sha256": "application",
        "validation_sha256": "validation",
    }


def _baseline_application_report(actor):
    """Provide two actor records so drift in a later actor cannot pass unnoticed."""
    return {
        "records": 2,
        "input_hash": "input",
        "model_revision": "model",
        "source_sha256": "left",
        "application_sha256": "application",
        "validation_sha256": "validation",
        "udf_sha256": "udf",
        "processed_hash": "left-images",
        "mean_embedding": [0.1, 0.2],
        "model_initializations": [copy.deepcopy(actor), copy.deepcopy(actor)],
    }


@pytest.fixture
def comparison_reports():
    """Provide baseline, changed and restored reports with fixed model provenance.

    Args:
        None.
    Returns:
        Baseline, changed and restored report dictionaries.
    Raises:
        None.
    """
    baseline = _baseline_application_report(_baseline_model_provenance())
    changed = copy.deepcopy(baseline)
    changed.update(source_sha256="right", processed_hash="right-images", mean_embedding=[0.4, 0.1])
    for initialization in changed["model_initializations"]:
        initialization["source_sha256"] = "right"
    return baseline, changed, copy.deepcopy(baseline)


def test_comparison_requires_changed_inference_and_restoration(recipe, comparison_reports):
    """Comparison requires changed inference and restoration.

    Args:
        recipe: Isolated example modules and canonical source paths.
        comparison_reports: Baseline, changed and restored provenance reports.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: A contract assertion fails.
        pytest.fail.Exception: An expected exception is not raised.
    """
    base, changed, restored = comparison_reports
    assert recipe.validation.compare_reports(base, changed, restored)["changed_mean_embedding_l2"] > 0.01
    with pytest.raises(ValueError, match="meaningfully change"):
        recipe.validation.compare_reports(base, {**changed, "mean_embedding": base["mean_embedding"]}, base)
    with pytest.raises(ValueError, match="tolerance"):
        recipe.validation.compare_reports(base, changed, {**base, "mean_embedding": [0.1, 0.3]})


@pytest.mark.parametrize("report_index", [0, 1, 2])
@pytest.mark.parametrize("field", ["model_files", "model_config_sha256", "udf_sha256", "precision",
                                   "gpu_capability", "python", "ray", "torch", "cuda",
                                   "transformers", "pyarrow", "lancedb", "model_revision"])
def test_comparison_rejects_model_or_runtime_drift_in_every_job_actor(recipe, comparison_reports, report_index, field):
    """Comparison rejects model or runtime drift in every job actor.

    Args:
        recipe: Isolated example modules and canonical source paths.
        comparison_reports: Baseline, changed and restored provenance reports.
        report_index: Job report whose actor provenance is modified.
        field: Provenance field selected for the negative test.
    Returns:
        None after the assertions pass.
    Raises:
        pytest.fail.Exception: An expected exception is not raised.
    """
    actor = comparison_reports[report_index]["model_initializations"][1]
    original = actor[field]
    if isinstance(original, list):
        actor[field] = [*original, "changed"]
    else:
        actor[field] = original + "-changed"
    with pytest.raises(ValueError, match="changed actual model/runtime|Actor model revision|different.*source than submitted"):
        recipe.validation.compare_reports(*comparison_reports)


@pytest.mark.parametrize("field", ["application_sha256", "validation_sha256", "udf_sha256"])
def test_comparison_rejects_changing_other_application_modules(recipe, comparison_reports, field):
    """Comparison rejects changing other application modules.

    Args:
        recipe: Isolated example modules and canonical source paths.
        comparison_reports: Baseline, changed and restored provenance reports.
        field: Provenance field selected for the negative test.
    Returns:
        None after the assertions pass.
    Raises:
        pytest.fail.Exception: An expected exception is not raised.
    """
    comparison_reports[1][field] = "different-source"
    for actor in comparison_reports[1]["model_initializations"]:
        actor[field] = "different-source"
    with pytest.raises(ValueError, match="changed fixed input " + field):
        recipe.validation.compare_reports(*comparison_reports)


def test_comparison_rejects_missing_model_provenance(recipe, comparison_reports):
    """Comparison rejects missing model provenance.

    Args:
        recipe: Isolated example modules and canonical source paths.
        comparison_reports: Baseline, changed and restored provenance reports.
    Returns:
        None after the assertions pass.
    Raises:
        pytest.fail.Exception: An expected exception is not raised.
    """
    comparison_reports[1]["model_initializations"][1].pop("model_files")
    with pytest.raises(ValueError, match="lacks actual model/runtime provenance"):
        recipe.validation.compare_reports(*comparison_reports)


@pytest.mark.parametrize("filename,field", [("application.py", "application_sha256"),
                                           ("validation.py", "validation_sha256"),
                                           ("npa_lancedb_bdd100k_udfs.py", "udf_sha256")])
@pytest.mark.parametrize("target", ["driver", "replacement_actor"])
def test_report_rejects_unsubmitted_driver_or_replacement_actor_import(
    recipe, source_bundle, tmp_path, filename, field, target
):
    """Report rejects unsubmitted driver or replacement actor import.

    Args:
        recipe: Isolated example modules and canonical source paths.
        source_bundle: Fixture that copies the ordinary Jobs source files.
        tmp_path: Temporary directory owned by this test.
        filename: Submitted source filename that must remain unchanged.
        field: Provenance field selected for the negative test.
        target: Driver or replacement actor selected for tampering.
    Returns:
        None after the assertions pass.
    Raises:
        pytest.fail.Exception: An expected exception is not raised.
    """
    manifest = source_bundle(tmp_path)
    report = {field: manifest[name] for name, field in recipe.validation.SOURCE_HASH_FIELDS.items()}
    report["model_initializations"] = [dict(report), dict(report)]
    recipe.validation.verify_submitted_sources(report, manifest)
    if target == "driver":
        report[field] = "unsubmitted-source"
    else:
        report["model_initializations"][1][field] = "unsubmitted-source"
    with pytest.raises(ValueError, match=filename):
        recipe.validation.verify_submitted_sources(report, manifest)


def test_comparison_checks_every_persisted_vector_even_when_means_match(recipe, tmp_path):
    """Comparison checks every persisted vector even when means match.

    Args:
        recipe: Isolated example modules and canonical source paths.
        tmp_path: Temporary directory owned by this test.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: A contract assertion fails.
        pytest.fail.Exception: An expected exception is not raised.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = [tmp_path / name for name in ("baseline.parquet", "changed.parquet")]
    for path, vectors in zip(paths, ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]), strict=True):
        pq.write_table(pa.table({"record_id": [0, 1], "input_sha256": ["first", "second"], "vector": vectors}), path)
    assert recipe.validation.compare_vectors(*paths, changed=True)["compared_vectors"] == 2
    with pytest.raises(ValueError, match="Restored persisted vectors"):
        recipe.validation.compare_vectors(*paths, changed=False)


def test_barrier_uses_one_coordinator_clock_and_observes_overlap(recipe):
    """Barrier uses one coordinator clock and observes overlap.

    Args:
        recipe: Isolated example modules and canonical source paths.
    Returns:
        None after the assertions pass.
    Raises:
        AssertionError: The tested behavior differs from its required contract.
    """
    async def exercise():
        """Verify that one coordinator clock observes overlapping actor calls.

        Args:
            None.
        Returns:
            None after the assertions pass.
        Raises:
            AssertionError: The tested behavior differs from its required contract.
        """
        barrier = recipe.application.InferenceBarrier(2)
        await asyncio.gather(barrier.arrive("first"), barrier.arrive("second"))
        await barrier.start("first")
        await barrier.start("second")
        await barrier.finish("first")
        await barrier.finish("second")
        result = await barrier.status()
        assert result["participants"] == 2 and result["overlap"]
        assert [item["monotonic_ns"] for item in result["events"]] == sorted(item["monotonic_ns"] for item in result["events"])
    asyncio.run(exercise())


def test_aggregation_rejects_a_removed_commit_marker(recipe, tmp_path):
    """Reject real Parquet bytes after their authoritative commit marker disappears.

    Args:
        recipe: Isolated application and validation modules.
        tmp_path: Driver-owned test output directory.
    Returns:
        None after the missing-commit regression is rejected.
    Raises:
        pytest.fail.Exception: Aggregation accepts the uncommitted shard.
    """
    import pyarrow

    shard = recipe.worker.preprocess_shard([0])
    vector = [1.0] + [0.0] * 511
    vectors = pyarrow.array([vector], type=pyarrow.list_(pyarrow.float32(), 512))
    directory = tmp_path / "shards" / "000000"
    receipt = recipe.application.commit_shard(
        directory, shard, vectors, {"inference_seconds": 0.1}, "revision", "execution")
    (directory / "commit.json").unlink()
    with pytest.raises(ValueError, match="commit marker"):
        recipe.application.aggregate(tmp_path, [receipt], 1)


def _unexpected_cluster_connection(**arguments):
    """Fail before a test can accidentally select any real Ray cluster."""
    pytest.fail("Invalid application GCS address reached ray.init")


@pytest.mark.parametrize("address", ["", ":6381", "auto", "127.0.0.1:6380", "http://127.0.0.1:6381"])
def test_entrypoint_rejects_invalid_application_gcs_before_connecting(recipe, tmp_path, monkeypatch, address):
    """Prevent invalid or management-runtime addresses from reaching Ray initialization.

    Args:
        recipe: Isolated example modules.
        tmp_path: Driver-owned test output directory.
        monkeypatch: Fixture restoring runtime module and environment changes.
        address: Invalid application GCS address supplied by the case.
    Returns:
        None after the public entrypoint rejects the address.
    Raises:
        pytest.fail.Exception: The address reaches Ray or is accepted.
    """
    ray = SimpleNamespace(init=_unexpected_cluster_connection, shutdown=lambda: None)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv("RAY_ADDRESS", address)
    with pytest.raises(ValueError, match="GCS port 6381"):
        recipe.application.main(["--output-path", str(tmp_path)])
