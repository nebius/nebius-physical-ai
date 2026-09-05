"""Hermetic contracts for the real Ray CLIP application; GPU execution is live."""

from __future__ import annotations

import asyncio
import copy
import importlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def recipe(monkeypatch):
    directory = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"
    monkeypatch.syspath_prepend(str(directory))
    names = ("validation", "worker", "submit", "application")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    modules = {name: importlib.import_module(name) for name in names}
    udf_source = Path(__file__).parents[2] / "src/npa/workbench/lancedb/bdd100k_udfs.py"
    yield SimpleNamespace(**modules, directory=directory, udf_source=udf_source)
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def test_source_revision_changes_real_pixels_and_restores_source(recipe, tmp_path):
    from PIL import Image

    manifests = {revision: recipe.submit.prepare_source(recipe.directory, tmp_path / revision, revision, udf_source=recipe.udf_source)
                 for revision in ("baseline", "changed", "restored")}
    assert manifests["baseline"] == manifests["restored"]
    assert manifests["changed"]["worker.py"] != manifests["baseline"]["worker.py"]
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


def test_source_upload_refuses_symlinks_and_keeps_unlisted_files_out(recipe, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in recipe.submit.SOURCE_FILES:
        (source / name).write_bytes((recipe.directory / name).read_bytes())
    (source / "operator.env").write_text("must never be uploaded")
    destination = tmp_path / "upload"
    recipe.submit.prepare_source(source, destination, "baseline", udf_source=recipe.udf_source)
    assert sorted(path.name for path in destination.iterdir()) == sorted((*recipe.submit.SOURCE_FILES, recipe.submit.UDF_FILENAME))
    (source / "worker.py").unlink()
    (source / "worker.py").symlink_to(recipe.directory / "worker.py")
    with pytest.raises(ValueError, match="symbolic links"):
        recipe.submit.prepare_source(source, tmp_path / "rejected", "baseline", udf_source=recipe.udf_source)


def test_source_bundle_contains_exact_explicit_workbench_udf_bytes(recipe, tmp_path):
    original = recipe.udf_source.read_bytes()
    for revision in ("baseline", "changed", "restored"):
        destination = tmp_path / revision
        manifest = recipe.submit.prepare_source(recipe.directory, destination, revision, udf_source=recipe.udf_source)
        assert (destination / recipe.submit.UDF_FILENAME).read_bytes() == original
        assert manifest[recipe.submit.UDF_FILENAME] == recipe.validation.file_hash(recipe.udf_source)
    assert recipe.udf_source.read_bytes() == original
    link = tmp_path / "linked-udf.py"
    link.symlink_to(recipe.udf_source)
    with pytest.raises(ValueError, match="symbolic links"):
        recipe.submit.prepare_source(recipe.directory, tmp_path / "rejected", "baseline", udf_source=link)


def test_application_imports_real_udf_from_jobs_bundle(recipe, tmp_path, monkeypatch):
    import pyarrow as pa

    recipe.submit.prepare_source(recipe.directory, tmp_path, "baseline", udf_source=recipe.udf_source)
    monkeypatch.setattr(recipe.application, "__file__", "/unavailable-driver/ray-session/application.py")
    monkeypatch.setattr(recipe.worker, "__file__", str(tmp_path / "worker.py"))
    monkeypatch.setitem(sys.modules, "npa_lancedb_bdd100k_udfs", None)
    udf = recipe.application.load_workbench_udf()
    assert Path(udf.__file__).resolve() == (tmp_path / recipe.submit.UDF_FILENAME).resolve()
    assert recipe.validation.file_hash(Path(udf.__file__)) == recipe.validation.file_hash(recipe.udf_source)
    batch = pa.record_batch({"image_bytes": [recipe.worker.render_record(42)]})
    assert udf.udf_dhash(batch).type == pa.int64()
    (tmp_path / recipe.submit.UDF_FILENAME).unlink()
    with pytest.raises(RuntimeError, match="lacks a regular Workbench CLIP UDF"):
        recipe.application.load_workbench_udf()


def test_actor_provenance_uses_worker_bundle_when_driver_path_is_unavailable(recipe, tmp_path, monkeypatch):
    import importlib.metadata

    bundle = tmp_path / "worker-runtime-env"
    manifest = recipe.submit.prepare_source(recipe.directory, bundle, "baseline", udf_source=recipe.udf_source)
    monkeypatch.setattr(recipe.application, "__file__", "/unavailable-driver/ray-session/application.py")
    monkeypatch.setattr(recipe.worker, "__file__", str(bundle / "worker.py"))
    monkeypatch.setattr(recipe.validation, "__file__", str(bundle / "validation.py"))
    real_loader = recipe.application.load_workbench_udf
    loads = []

    def load_udf():
        udf = real_loader()
        udf._clip_components = lambda **kwargs: loads.append(kwargs)
        return udf

    monkeypatch.setattr(recipe.application, "load_workbench_udf", load_udf)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        __version__="2.46.0", get_runtime_context=lambda: SimpleNamespace(
            get_node_id=lambda: "worker-node", get_accelerator_ids=lambda: {"GPU": ["0"]})))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="test-version", version=SimpleNamespace(cuda="test-version"), cuda=SimpleNamespace(
            is_available=lambda: True, synchronize=lambda: None, get_device_name=lambda index: "test-gpu",
            get_device_capability=lambda index: (12, 0))))
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"clip"}')
    (checkpoint / "pytorch_model.bin").write_bytes(b"test weights; no inference in this unit test")

    actor = recipe.application.ClipActor(str(checkpoint), "test-model-revision")
    assert loads == [{"device": "cuda:0", "precision": "float32"}]
    assert actor.info["application_sha256"] == manifest["application.py"]
    assert actor.info["source_sha256"] == manifest["worker.py"]
    assert actor.info["validation_sha256"] == manifest["validation.py"]
    assert actor.info["udf_sha256"] == manifest[recipe.submit.UDF_FILENAME]
    assert Path(actor.info["udf_module_path"]).parent == bundle
    assert actor.udf.CLIP_MODEL_NAME == str(checkpoint.resolve())


@pytest.mark.parametrize("ids,count", [([0, 0], 2), ([0, 2], 2), ([1], 1)])
def test_completeness_rejects_duplicate_missing_and_extra_ids(recipe, ids, count):
    with pytest.raises(ValueError, match="record IDs"):
        recipe.validation.verify_ids(ids, count)


def test_partition_covers_uneven_batches(recipe):
    chunks = recipe.validation.partitions(131, 64)
    assert list(map(len, chunks)) == [64, 64, 3]
    recipe.validation.verify_ids([item for chunk in chunks for item in chunk], 131)


def test_checkpoint_replay_skips_actor_and_corruption_fails(recipe, tmp_path):
    shard = recipe.worker.preprocess_shard([0, 1])
    identity = recipe.validation.checkpoint_identity(shard, "revision", "execution")
    data = tmp_path / "embeddings.parquet"
    data.write_bytes(b"checkpoint bytes")
    receipt = {"identity": identity, "parquet_sha256": recipe.validation.file_hash(data)}
    recipe.validation.atomic_json(tmp_path / "commit.json", receipt)
    forbidden_actor = SimpleNamespace(infer=SimpleNamespace(remote=lambda *args: pytest.fail("replay dispatched GPU work")))
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
    recipe.validation.verify_execution(tmp_path, "first")
    recipe.validation.verify_execution(tmp_path, "first")
    with pytest.raises(ValueError, match="different execution fingerprint"):
        recipe.validation.verify_execution(tmp_path, "different")
    (tmp_path / "execution.json").unlink()
    (tmp_path / "old-shard").write_bytes(b"old")
    with pytest.raises(ValueError, match="no execution fingerprint"):
        recipe.validation.verify_execution(tmp_path, "first")


def test_model_fingerprint_ignores_download_metadata_but_covers_model_bytes(recipe, tmp_path):
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
    calls = []
    def kill(actor, **kwargs):
        calls.append(actor)
        if actor == "first":
            raise RuntimeError("actor unavailable")
    ray = SimpleNamespace(kill=kill, shutdown=lambda: calls.append("shutdown"))
    errors = recipe.application.cleanup_actors(ray, ["first", "second"])
    assert calls == ["first", "second", "shutdown"]
    assert errors == [{"operation": "kill actor", "actor_index": 0, "error_type": "RuntimeError"}]


def test_uncommitted_checkpoint_is_recomputed(recipe, tmp_path):
    (tmp_path / "embeddings.parquet").write_bytes(b"partial without commit marker")
    shard = recipe.worker.preprocess_shard([4])
    calls = []
    actor = SimpleNamespace(infer=SimpleNamespace(remote=lambda *args: calls.append(args) or "future"))
    cached, future = recipe.application.submit_shard(actor, shard, tmp_path, "revision", "execution")
    assert cached is None and future == "future" and len(calls) == 1


def test_real_parquet_checkpoint_lance_aggregation_and_retrieval(recipe, tmp_path):
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


def test_client_overrides_ambient_management_address_and_restores_it(recipe, monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:6380")
    with recipe.submit.application_address("http://127.0.0.1:18265"):
        assert recipe.submit.os.environ["RAY_ADDRESS"] == "http://127.0.0.1:18265"
    assert recipe.submit.os.environ["RAY_ADDRESS"] == "127.0.0.1:6380"


@pytest.mark.parametrize("missing", ["--app-address", "--udf-source"])
def test_client_requires_explicit_application_address_and_udf_before_importing_ray(recipe, capsys, missing):
    argv = ["--address", "http://127.0.0.1:18265", "--model-path", "/models/clip",
            "--model-revision", "revision", "--output-path", "/outputs/run", "--evidence-dir", "/private/evidence"]
    argv += (["--udf-source", str(recipe.udf_source)] if missing == "--app-address"
             else ["--app-address", "127.0.0.1:6381"])
    with pytest.raises(SystemExit) as exc:
        recipe.submit.main(argv)
    assert exc.value.code == 2
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize("address", ["http://127.0.0.1:8266", "http://public.example:8265", "http://secret@127.0.0.1:8265", "http://127.0.0.1:8265?token=secret"])
def test_client_rejects_management_or_exposed_endpoints(recipe, address):
    with pytest.raises(ValueError):
        with recipe.submit.application_address(address):
            pytest.fail("unsafe Jobs endpoint accepted")


@pytest.fixture
def comparison_reports():
    actor = {"model_revision": "model", "model_files": [{"path": "weights.bin", "sha256": "weights"}],
             "model_config_sha256": "config", "udf_sha256": "udf", "precision": "float32",
             "gpu_capability": [10, 0], "python": "3.12", "ray": "2.46.0", "torch": "2.7.0",
             "cuda": "12.8", "transformers": "4.49.0", "pyarrow": "19.0.0", "lancedb": "0.21.0",
             "source_sha256": "left", "application_sha256": "application", "validation_sha256": "validation"}
    base = {"records": 2, "input_hash": "input", "model_revision": "model", "source_sha256": "left",
            "application_sha256": "application", "validation_sha256": "validation", "udf_sha256": "udf",
            "processed_hash": "left-images", "mean_embedding": [0.1, 0.2],
            "model_initializations": [copy.deepcopy(actor), copy.deepcopy(actor)]}
    changed = copy.deepcopy(base)
    changed.update(source_sha256="right", processed_hash="right-images", mean_embedding=[0.4, 0.1])
    for initialization in changed["model_initializations"]:
        initialization["source_sha256"] = "right"
    return base, changed, copy.deepcopy(base)


def test_comparison_requires_changed_inference_and_restoration(recipe, comparison_reports):
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
    actor = comparison_reports[report_index]["model_initializations"][1]
    original = actor[field]
    actor[field] = [*original, "changed"] if isinstance(original, list) else original + "-changed"
    with pytest.raises(ValueError, match="changed actual model/runtime|Actor model revision|different.*source than submitted"):
        recipe.validation.compare_reports(*comparison_reports)


@pytest.mark.parametrize("field", ["application_sha256", "validation_sha256", "udf_sha256"])
def test_comparison_rejects_changing_other_application_modules(recipe, comparison_reports, field):
    comparison_reports[1][field] = "different-source"
    for actor in comparison_reports[1]["model_initializations"]:
        actor[field] = "different-source"
    with pytest.raises(ValueError, match="changed fixed input " + field):
        recipe.validation.compare_reports(*comparison_reports)


def test_comparison_rejects_missing_model_provenance(recipe, comparison_reports):
    comparison_reports[1]["model_initializations"][1].pop("model_files")
    with pytest.raises(ValueError, match="lacks actual model/runtime provenance"):
        recipe.validation.compare_reports(*comparison_reports)


@pytest.mark.parametrize("filename,field", [("application.py", "application_sha256"),
                                           ("validation.py", "validation_sha256"),
                                           ("npa_lancedb_bdd100k_udfs.py", "udf_sha256")])
@pytest.mark.parametrize("target", ["driver", "replacement_actor"])
def test_completed_job_rejects_unsubmitted_application_or_validation_import(recipe, tmp_path, filename, field, target):
    class Client:
        def submit_job(self, **kwargs):
            self.submission = kwargs
            directory = Path(kwargs["runtime_env"]["working_dir"])
            report = {key: recipe.validation.file_hash(directory / name)
                      for name, key in recipe.validation.SOURCE_HASH_FIELDS.items()}
            report["model_initializations"] = [dict(report), dict(report)]
            if target == "driver":
                report[field] = "unsubmitted-source"
            else:
                report["model_initializations"][1][field] = "unsubmitted-source"
            self.logs = "RAY_CLIP_REPORT " + json.dumps(report)
            return kwargs["submission_id"]

        def get_job_info(self, identity):
            return SimpleNamespace(status="SUCCEEDED", start_time=1, end_time=2,
                                   runtime_env=self.submission["runtime_env"])

        def get_job_logs(self, identity):
            return self.logs

    args = SimpleNamespace(output_path="/outputs/run", python="/app/env/bin/python", model_path="/models/clip",
                           model_revision="revision", records=128, actors=1, batch_size=64,
                           udf_source=str(recipe.udf_source),
                           recovery_check=False, evidence_dir=str(tmp_path), app_address="127.0.0.1:6381")
    with pytest.raises(ValueError, match=filename):
        recipe.submit.run_job(Client(), args, recipe.directory, "baseline")


def test_comparison_checks_every_persisted_vector_even_when_means_match(recipe, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = [tmp_path / name for name in ("baseline.parquet", "changed.parquet")]
    for path, vectors in zip(paths, ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]), strict=True):
        pq.write_table(pa.table({"record_id": [0, 1], "input_sha256": ["first", "second"], "vector": vectors}), path)
    assert recipe.validation.compare_vectors(*paths, changed=True)["compared_vectors"] == 2
    with pytest.raises(ValueError, match="Restored persisted vectors"):
        recipe.validation.compare_vectors(*paths, changed=False)


def test_barrier_uses_one_coordinator_clock_and_observes_overlap(recipe):
    async def exercise():
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


def test_cancel_targets_exact_submission_and_waits_for_terminal(recipe, tmp_path, monkeypatch):
    monkeypatch.setattr(recipe.submit.time, "sleep", lambda _: None)
    class Client:
        def submit_job(self, **kwargs):
            self.submission = kwargs
            self.stopped = False
            assert set(Path(kwargs["runtime_env"]["working_dir"]).iterdir())
            return kwargs["submission_id"]

        def get_job_info(self, identity):
            assert identity == self.submission["submission_id"]
            return SimpleNamespace(status="STOPPED" if self.stopped else "RUNNING", start_time=1, end_time=2,
                                   runtime_env=self.submission["runtime_env"])

        def get_job_logs(self, identity):
            assert identity == self.submission["submission_id"]
            return "RAY_CLIP_FIRST_CHECKPOINT\n"

        def stop_job(self, identity):
            assert identity == self.submission["submission_id"]
            self.stopped = True
            return True

    args = SimpleNamespace(output_path="/outputs/run", python="/app/env/bin/python", model_path="/models/clip",
                           model_revision="revision", records=128, actors=1, batch_size=64,
                           udf_source=str(recipe.udf_source),
                           recovery_check=False, evidence_dir=str(tmp_path), app_address="127.0.0.1:6379")
    client = Client()
    result = recipe.submit.run_job(client, args, recipe.directory, "baseline", cancel=True)
    assert result["status"] == "STOPPED" and result["stop_requested"]
    assert [row["status"] for row in result["status_observations"]] == ["RUNNING", "STOPPED"]
    saved = json.loads((tmp_path / f'{result["submission_id"]}.json').read_text())
    assert saved["status"] == "STOPPED"
