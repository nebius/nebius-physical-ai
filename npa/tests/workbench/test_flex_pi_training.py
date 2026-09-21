"""Public contract and durable checkpoint integrity gates, without vendor imports."""

import json
import hashlib
from pathlib import Path
import shutil

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.storage import StorageError
from npa.sdk.workbench.flex_pi import train
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training_artifacts import publish_checkpoint, restore_checkpoint
from npa.workbench.flex_pi.training_assets import _prepare_derived
from npa.workbench.flex_pi.training_worker import _workload_identity


def test_cli_and_sdk_resolve_the_same_immutable_public_contract():
    result = CliRunner().invoke(app, ["workbench", "flex-pi", "train", "--dry-run",
                                     "--output-path", "s3://example-bucket/run/train"])
    assert result.exit_code == 0, result.output
    cli = json.loads(result.stdout)
    assert cli == train(output_path="s3://example-bucket/run/train", dry_run=True)
    assert cli["gpu_count"] == 4
    assert cli["microbatch_per_rank"] * cli["gradient_accumulation_steps"] * 4 == 96
    assert cli["train_frames"] == 115620
    assert cli["validation_frames"] == 12390
    assert cli["final_training_batch"] == 36
    assert cli["non_comparable_to_reference"] is True
    assert cli["reference_benchmark_beaten"] is False


@pytest.mark.parametrize("destination", ["", "s3://example-bucket", "/tmp/train",
                                        "s3://example-bucket/run?token=hidden",
                                        "s3://user:secret@example-bucket/run"])
def test_invalid_training_destinations_fail_before_execution(destination):
    with pytest.raises(FlexPiError):
        train(output_path=destination, dry_run=True)


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, uri):
        self.objects[uri] = Path(path).read_bytes()

    def download_file(self, uri, path):
        Path(path).write_bytes(self.objects[uri])


def test_checkpoint_readback_requires_every_original_byte(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "trainer_state.json").write_text('{"global_step":1205}')
    (source / "optimizer.bin").write_bytes(b"generated optimizer state")
    for name in ("model.safetensors", "scheduler.bin", *(f"random_states_{rank}.pkl" for rank in range(4))):
        (source / name).write_bytes(b"generated " + name.encode())
    storage = MemoryStorage()
    prefix = "s3://example-bucket/run/checkpoint"
    manifest = publish_checkpoint(source, prefix, storage)
    restore_checkpoint(manifest, prefix, tmp_path / "restored", storage)
    assert (tmp_path / "restored/optimizer.bin").read_bytes() == (source / "optimizer.bin").read_bytes()
    storage.objects[prefix + "/optimizer.bin"] = b"tampered optimizer state"
    with pytest.raises(FlexPiError, match="content verification"):
        restore_checkpoint(manifest, prefix, tmp_path / "tampered", storage)
    shutil.rmtree(source)
    assert len(manifest["files"]) == 8


def test_checkpoint_manifest_cannot_escape_private_restore_root(tmp_path):
    manifest = {"files": [{"path": "../outside", "size": 0, "sha256": "0" * 64}]}
    with pytest.raises(StorageError):
        restore_checkpoint(manifest, "s3://example-bucket/run/checkpoint", tmp_path / "restore", MemoryStorage())
    assert not (tmp_path / "outside").exists()


def test_incomplete_rank_checkpoint_cannot_be_published(tmp_path):
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "trainer_state.json",
                 *(f"random_states_{rank}.pkl" for rank in range(3))):
        (tmp_path / name).write_bytes(b"incomplete generated state")
    storage = MemoryStorage()
    with pytest.raises(FlexPiError, match="four-rank RNG"):
        publish_checkpoint(tmp_path, "s3://example-bucket/run/checkpoint", storage)
    assert not storage.objects


@pytest.mark.parametrize("changed_inputs", [False, True])
def test_derived_cache_requires_input_binding_and_prior_receipt(tmp_path, monkeypatch, changed_inputs):
    action = tmp_path / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    action.write_bytes(b"derived initialization")
    (tmp_path / "derived-initialization.json").write_text(json.dumps({
        "sha256": hashlib.sha256(action.read_bytes()).hexdigest(),
        "source_manifest_sha256": "old" if changed_inputs else "current",
    }))
    text = tmp_path / "text"
    text.mkdir()
    (text / "embedding.pt").write_bytes(b"unreceipted text embedding")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: pytest.fail("must fail before preprocessing"))
    with pytest.raises(RuntimeError, match="changed" if changed_inputs else "unreceipted"):
        _prepare_derived(tmp_path, "current")


def test_workload_identity_ignores_execution_paths_but_preserves_model_and_normalization(tmp_path):
    identities = []
    for name in ("profile", "train", "resume"):
        root = tmp_path / name
        root.mkdir()
        (root / "dataset_stats.json").write_text('{"mean":1}')
        assets = root / "assets"
        plan = {"configuration": {"output_dir": str(root), "resume": str(root / "state"),
                                   "num_workers": len(name), "model": {"width": 32},
                                   "data": {"path": str(assets / "dataset")}}}
        identities.append(_workload_identity(plan, root, assets))
    assert identities[0] == identities[1] == identities[2]
    plan["configuration"]["model"]["width"] = 64
    assert _workload_identity(plan, root, assets)["workload_sha256"] != identities[0]["workload_sha256"]
    (root / "dataset_stats.json").write_text('{"mean":2}')
    assert _workload_identity(plan, root, assets)["normalization_sha256"] != identities[0]["normalization_sha256"]


def test_live_optimizer_digest_detects_state_changes_after_restoration():
    import copy
    import torch
    from npa.workbench.flex_pi.training_state import state_digest

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    parameter.square().sum().backward()
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    original = state_digest(saved)
    restored = torch.optim.AdamW([parameter], lr=1e-4)
    restored.load_state_dict(saved)
    assert state_digest(restored.state_dict()) == original
    restored.state[parameter]["exp_avg"][0] += 1
    assert state_digest(restored.state_dict()) != original


def test_training_state_digest_preserves_dictionary_boundaries():
    from npa.workbench.flex_pi.training_state import state_digest

    assert state_digest({"a": {"b": 1}, "c": 2}) != state_digest({"a": {"b": 1, "c": 2}})
