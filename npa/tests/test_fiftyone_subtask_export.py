"""Exercise native-export identity mapping and shared storage without services."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.cli.fiftyone.subtasks import _subtask_export_python_script
from npa.clients.storage import StorageClient, StorageError
from npa.fiftyone_lerobot_subtasks import (
    SubtaskLabelError,
    _upload_directory,
    export_fiftyone_subtasks,
    export_fiftyone_subtasks_to_s3,
)


class Episode(dict):
    def __init__(self, index):
        super().__init__(episode_index=index)
        self.id = f"sample-{index}"


def _write_native_export(root, samples):
    root = Path(root)
    (root / "meta").mkdir(exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 10, "features": {}}))
    rows = [
        {"episode_index": output, "frame_index": frame, "timestamp": frame / 10,
         "action": [float(sample["episode_index"])]}
        for output, sample in enumerate(samples) for frame in range(2)
    ]
    (root / "data").mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "data/frames.parquet")


@pytest.fixture
def native_dataset(monkeypatch):
    samples = [Episode(7), Episode(2)]
    tags = {
        str(index): {"sample_id": sample.id, "start": 0, "end": 200_000_000,
                     "tag": f"subtask:source-{sample['episode_index']}", "index_type": 2}
        for index, sample in enumerate(samples)
    }
    dataset = Mock(media_type="multimodal", temporal_tags=tags)
    dataset.iter_samples.return_value = iter(samples)

    def select(ids, *, ordered):
        assert ordered is True
        selected = [next(sample for sample in samples if sample.id == id_) for id_ in ids]
        view = Mock()
        view.export.side_effect = lambda *, export_dir, dataset_type: _write_native_export(export_dir, selected)
        return view

    dataset.select.side_effect = select
    fo = SimpleNamespace(types=SimpleNamespace(LeRobotDataset=object()),
                         list_datasets=lambda: ["review"], load_dataset=lambda name: dataset)
    monkeypatch.setitem(sys.modules, "fiftyone", fo)
    return dataset, samples


def test_export_freezes_order_and_reindexes_noncontiguous_source_episodes(native_dataset, tmp_path):
    dataset, _samples = native_dataset
    report = export_fiftyone_subtasks("review", tmp_path / "derived")
    rows = pq.read_table(tmp_path / "derived/data/frames.parquet").to_pylist()
    catalog = pq.read_table(tmp_path / "derived/meta/subtasks.parquet").to_pylist()
    labels = {row["subtask_index"]: row["subtask"] for row in catalog}
    assert [(row["episode_index"], row["action"][0], labels[row["subtask_index"]]) for row in rows] == [
        (0, 2.0, "source-2"), (0, 2.0, "source-2"), (1, 7.0, "source-7"), (1, 7.0, "source-7"),
    ]
    dataset.select.assert_called_once_with(["sample-2", "sample-7"], ordered=True)
    dataset.export.assert_not_called()
    assert report["episode_index_mapping"] == [
        {"source_episode_index": 2, "export_episode_index": 0},
        {"source_episode_index": 7, "export_episode_index": 1},
    ]


@pytest.mark.parametrize("index", [None, -1, True, "7", 2])
def test_export_rejects_invalid_or_duplicate_episode_identity(native_dataset, tmp_path, index):
    dataset, samples = native_dataset
    samples[0]["episode_index"] = index
    with pytest.raises(SubtaskLabelError, match="episode"):
        export_fiftyone_subtasks("review", tmp_path / "derived")
    dataset.select.assert_not_called()
    assert not (tmp_path / "derived").exists()


@pytest.fixture
def s3(monkeypatch):
    client = Mock()
    client.list_objects_v2.return_value = {}
    factory = Mock(return_value=client)
    monkeypatch.setattr("boto3.client", factory)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example.invalid")
    return client, factory


def test_export_to_s3_uses_retry_configured_shared_storage(native_dataset, s3):
    client, factory = s3
    report = export_fiftyone_subtasks_to_s3("review", "s3://bucket/derived")
    assert report["uploaded_files"] == 4
    assert "output_dir" not in report
    assert report["output_path"] == "s3://bucket/derived"
    client.list_objects_v2.assert_called_once_with(Bucket="bucket", Prefix="derived/", MaxKeys=1)
    assert client.upload_file.call_count == 4
    assert factory.call_args.kwargs["config"].retries == {"max_attempts": 3, "mode": "adaptive"}


def test_remote_bundle_executes_shared_storage_without_installed_npa(native_dataset, s3, monkeypatch, capsys, tmp_path):
    import builtins

    # Execute the trusted generated entrypoint with normal __main__ script semantics.
    script_path = tmp_path / "remote_export.py"
    script_path.write_text(_subtask_export_python_script("review", "s3://bucket/derived/"), encoding="utf-8")
    original_import = builtins.__import__

    def without_npa(name, *args, **kwargs):
        if name == "npa" or name.startswith("npa."):
            raise ModuleNotFoundError(name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_npa)
    runpy.run_path(str(script_path), run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert report["episode_count"] == 2
    assert report["uploaded_files"] == 4
    assert s3[1].call_args.kwargs["config"].retries["mode"] == "adaptive"


def test_nonempty_prefix_never_uploads(tmp_path, s3):
    client, _factory = s3
    client.list_objects_v2.return_value = {"Contents": [{"Key": "derived/existing"}]}
    (tmp_path / "file").write_text("new")
    with pytest.raises(StorageError, match="must be empty"):
        _upload_directory(tmp_path, "s3://bucket/derived/")
    client.upload_file.assert_not_called()


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_refusal_precedes_any_upload(tmp_path, s3, directory):
    target = tmp_path / "target"
    target.mkdir() if directory else target.write_text("bytes")
    (tmp_path / "link").symlink_to(target, target_is_directory=directory)
    with pytest.raises(SubtaskLabelError, match="symlinked"):
        _upload_directory(tmp_path, "s3://bucket/derived/")
    s3[1].assert_not_called()


@pytest.mark.parametrize("uri", ["s3:///missing-bucket", "file:///tmp/derived", "s3://bucket/key?query=1", "s3://bucket/key#fragment"])
def test_shared_storage_rejects_invalid_uri(tmp_path, s3, uri):
    with pytest.raises(StorageError, match="Expected s3:// URI"):
        _upload_directory(tmp_path, uri)
    s3[0].list_objects_v2.assert_not_called()
    s3[0].upload_file.assert_not_called()


@pytest.mark.parametrize("uri, prefix", [("s3://bucket", ""), ("s3://bucket/derived", "derived/"), ("s3://bucket/derived/", "derived/")])
def test_shared_storage_lists_and_uploads_the_same_prefix(tmp_path, s3, uri, prefix):
    (tmp_path / "file").write_text("bytes")
    storage = StorageClient.from_environment()
    storage.upload_directory(str(tmp_path), uri, require_empty=True)
    s3[0].list_objects_v2.assert_called_once_with(Bucket="bucket", Prefix=prefix, MaxKeys=1)
    s3[0].upload_file.assert_called_once_with(str(tmp_path / "file"), "bucket", prefix + "file")
