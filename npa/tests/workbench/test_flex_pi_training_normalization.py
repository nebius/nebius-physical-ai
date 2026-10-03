"""Keep the original normalization bytes across independent jobs and resume."""

import hashlib
import json
from pathlib import Path

import pytest

from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training import TrainingRequest, _assert_resume_parity
from npa.workbench.flex_pi.training_artifacts import (
    publish_checkpoint,
    restore_checkpoint,
)
from npa.workbench.flex_pi.training_normalization import (
    normalization_overrides,
    stage_normalization,
    validate_normalization,
)
from npa.workbench.flex_pi.training_worker import _workload_identity


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, uri):
        self.objects[uri] = Path(path).read_bytes()

    def download_file(self, uri, path):
        Path(path).write_bytes(self.objects[uri])


@pytest.mark.parametrize(
    "path,digest",
    [
        ("s3://example-bucket/stats.json", ""),
        ("", "a" * 64),
        ("s3://example-bucket/stats.json?token=x", "a" * 64),
        ("s3://example-bucket/prefix/", "a" * 64),
        ("s3://example-bucket/stats.json", "not-a-digest"),
    ],
)
def test_normalization_requires_safe_paired_identity(path, digest):
    request = TrainingRequest(
        "s3://example-bucket/run", normalization_path=path, normalization_sha256=digest
    )
    with pytest.raises(FlexPiError, match="exact S3 object and SHA-256"):
        validate_normalization(request)


def test_normalization_download_rejects_changed_bytes(tmp_path):
    storage = MemoryStorage()
    source = "s3://example-bucket/original/dataset_stats.json"
    storage.objects[source] = b'{"q01":[0.0]}'
    request = TrainingRequest(
        "s3://example-bucket/run",
        normalization_path=source,
        normalization_sha256="a" * 64,
    )
    with pytest.raises(FlexPiError, match="content verification"):
        stage_normalization(request, tmp_path, storage)


def _workload(plan_root, normalization):
    config = {
        "output_dir": str(plan_root),
        "data": {
            split: {"pretrained_norm_stats": str(normalization)}
            for split in ("train", "val")
        },
    }
    return _workload_identity(
        {"configuration": config}, plan_root, plan_root / "assets"
    )


def _stage_original(tmp_path):
    storage = MemoryStorage()
    source = "s3://example-bucket/original/dataset_stats.json"
    payload = b'{\n  "q01": [0.125],\n  "q99": [0.75]\n}'
    storage.objects[source] = payload
    request = TrainingRequest(
        "s3://example-bucket/run",
        normalization_path=source,
        normalization_sha256=hashlib.sha256(payload).hexdigest(),
    )
    original = tmp_path / "original"
    original.mkdir()
    stats = stage_normalization(request, original, storage)
    return storage, request, payload, original, stats


def _restore_state(storage, original, stats, prefix, restored):
    checkpoint = original / "checkpoint"
    checkpoint.mkdir()
    for name in (
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "trainer_state.json",
    ):
        (checkpoint / name).write_bytes(b"generated training state")
    for rank in range(4):
        (checkpoint / f"random_states_{rank}.pkl").write_bytes(b"generated RNG state")
    (checkpoint / "dataset_stats.json").write_bytes(stats.read_bytes())
    manifest = publish_checkpoint(checkpoint, prefix, storage)
    restore_checkpoint(manifest, prefix, restored, storage)


def test_checkpoint_freezes_normalization_and_resume_binds_its_bytes(tmp_path):
    storage, request, payload, original, stats = _stage_original(tmp_path)
    restored = tmp_path / "restored"
    _restore_state(
        storage, original, stats, request.output_path + "/checkpoint", restored
    )
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    overrides = normalization_overrides(
        {"normalization_file": str(restored / "dataset_stats.json")}, resumed
    )
    assert overrides == [
        f"data.{split}.pretrained_norm_stats="
        + json.dumps(str(resumed / "dataset_stats.json"))
        for split in ("train", "val")
    ]
    assert (resumed / "dataset_stats.json").read_bytes() == payload
    original_identity = _workload(original, stats)
    assert _workload(resumed, resumed / "dataset_stats.json") == original_identity
    (resumed / "dataset_stats.json").write_bytes(payload.replace(b"0.125", b"0.250"))
    with pytest.raises(FlexPiError, match="workload or normalization"):
        _assert_resume_parity(
            {**original_identity, "checkpoint": {}},
            _workload(resumed, resumed / "dataset_stats.json"),
        )
