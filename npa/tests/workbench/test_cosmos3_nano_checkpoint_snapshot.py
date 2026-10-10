"""Exercise immutable native checkpoint binding with explicit CPU-only native fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from npa.workbench.cosmos import nano_checkpoint_snapshot as snapshot

_REVISION = "a" * 40
_WAN_REVISION = "b" * 40


class FixtureTensor:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = tuple(shape or (len(values),))
        self.dtype = "fixture-float32"
        self.ndim = len(self.shape)

    def reshape(self, *_):
        return FixtureTensor(self.values)

    def contiguous(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def view(self, _):
        return self

    def numpy(self):
        return SimpleNamespace(
            tobytes=lambda: struct.pack(f"<{len(self.values)}f", *self.values)
        )


def _fixture_state(path):
    payload = json.loads(Path(path).read_text())
    return {
        name: FixtureTensor(item["values"], item["shape"])
        for name, item in payload.items()
    }


def _fixture_remap(name, _):
    return name.replace("decoder.block.0.", "decoder.layers.1.")


def _fixture_materializer(directory):
    root = Path(directory)
    source = _fixture_state(root / "diffusion_pytorch_model.safetensors")
    converted = {}
    for name, value in source.items():
        key = _fixture_remap(name, 1)
        shape = (len(value.values),) if key.endswith(".alpha") else value.shape
        converted[key] = {"shape": list(shape), "values": value.values}
    output = root / snapshot._SOUND_CHECKPOINT
    if not output.exists():
        output.write_text(json.dumps(converted))
    config = root / snapshot._SOUND_CONFIG
    if not config.exists():
        config.write_bytes((root / "config.json").read_bytes())


def _native_modules(monkeypatch, config):
    checkpoint_module = ModuleType("cosmos_framework.inference.common.checkpoints")
    checkpoint_module._materialize_avae_ckpt = _fixture_materializer
    checkpoint_module._avae_block_key_to_legacy = _fixture_remap
    monkeypatch.setitem(sys.modules, checkpoint_module.__name__, checkpoint_module)
    config_module = ModuleType("cosmos_framework.inference.common.config")
    config_module.unstructure_config = lambda value, invalid: copy.deepcopy(value)
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    torch = ModuleType("torch")
    torch.Tensor = FixtureTensor
    torch.uint8 = "fixture-uint8"
    torch.equal = lambda first, second: first.values == second.values
    torch.load = lambda path, **kwargs: {"state_dict": _fixture_state(path)}
    monkeypatch.setitem(sys.modules, "torch", torch)
    safetensors = ModuleType("safetensors.torch")
    safetensors.load_file = lambda path, device: _fixture_state(path)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors)
    monkeypatch.setattr(snapshot, "_config_path", lambda: config)


def _snapshot_fixture(root):
    weights = {
        "transformer.layer": "transformer/weights.safetensors",
        "vision_encoder.layer": "vision_encoder/model.safetensors",
    }
    files = {
        "config.json": '{"model":{}}',
        "model_index.json": '{"fixture":true}',
        "model.safetensors.index.json": json.dumps({"weight_map": weights}),
        "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
            {"weight_map": {"transformer.layer": "weights.safetensors"}}
        ),
        "transformer/weights.safetensors": "fixture-transformer-weights",
        "vision_encoder/model.safetensors": "fixture-vision-weights",
        "preprocessor_config.json": '{"fixture":true}',
        "tokenizer_config.json": '{"fixture":true}',
        "tokenizer.json": '{"fixture":true}',
        "sound_tokenizer/config.json": '{"fixture":true}',
        "sound_tokenizer/diffusion_pytorch_model.safetensors": json.dumps(
            {
                "decoder.block.0.snake1.alpha": {"shape": [1, 2, 1], "values": [1, 2]},
                "decoder.block.0.weight": {"shape": [2], "values": [3, 4]},
            }
        ),
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _remote_metadata(root, revision):
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        git_blob = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw, usedforsecurity=False
        ).hexdigest()
        name = str(path.relative_to(root))
        lfs = (
            None
            if name.endswith(".json")
            else SimpleNamespace(sha256=hashlib.sha256(raw).hexdigest())
        )
        entries.append(
            SimpleNamespace(rfilename=name, size=len(raw), blob_id=git_blob, lfs=lfs)
        )
    return SimpleNamespace(sha=revision, siblings=entries)


class FixtureHf:
    def __init__(self, repository, revision, subdirectory="", filename=""):
        self.repository = repository
        self.revision = revision
        self.subdirectory = subdirectory
        self.filename = filename

    def model_dump(self):
        return vars(self).copy()

    @classmethod
    def model_validate(cls, values):
        return cls(**values)


class FixtureCheckpoint:
    def __init__(self, hf, path, post_download=None):
        self.hf = hf
        self.path = path
        self.post_download = post_download

    def model_copy(self, update):
        copied = copy.copy(self)
        for name, value in update.items():
            setattr(copied, name, value)
        return copied

    def download(self):
        if self.post_download:
            assert self.hf.revision == _REVISION
            self.post_download(str(self.path))
        return str(self.path)


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    root = tmp_path / "native"
    nano = root / "hub" / "snapshots" / _REVISION
    _snapshot_fixture(nano)
    framework = root / "framework"
    for name in snapshot._SOURCES:
        source = framework / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture-source:" + name)
    wan = root / "wan" / "Wan2.2_VAE.pth"
    wan.parent.mkdir(parents=True)
    wan.write_text("fixture-wan-weights")
    _native_modules(monkeypatch, framework / snapshot._NANO_CONFIG)
    checkpoints = _checkpoint_fixtures(nano, wan)

    def directory_class(**kwargs):
        assert kwargs == {"repository": snapshot._REPOSITORY, "revision": _REVISION}
        return SimpleNamespace(download=lambda: str(nano))

    checkpoint_class = SimpleNamespace(from_uri=lambda uri: checkpoints[uri])
    monkeypatch.setattr(
        snapshot,
        "_native_checkpoint_support",
        lambda: (checkpoint_class, directory_class),
    )
    metadata = {
        snapshot._REPOSITORY: _remote_metadata(nano, _REVISION),
        "fixture/wan": _remote_metadata(wan.parent, _WAN_REVISION),
    }
    monkeypatch.setattr(
        snapshot, "_model_info", lambda repository, revision: metadata[repository]
    )
    return SimpleNamespace(
        nano=nano,
        framework=framework,
        wan=wan,
        metadata=metadata,
        checkpoints=checkpoints,
    )


def _checkpoint_fixtures(nano, wan):
    sound_hf = FixtureHf(snapshot._REPOSITORY, "main", "sound_tokenizer")
    wan_hf = FixtureHf("fixture/wan", _WAN_REVISION, filename=wan.name)
    return {
        snapshot._SOUND_URI: FixtureCheckpoint(
            sound_hf, nano / "sound_tokenizer", _fixture_materializer
        ),
        snapshot._WAN_URI: FixtureCheckpoint(wan_hf, wan),
    }


def _resolved_native(preparation):
    paths = preparation["local_paths"]
    setup = SimpleNamespace(
        checkpoint_type="hf",
        checkpoint_hf=None,
        use_torch_compile=False,
        config_file=paths["config_file"],
        download_checkpoint=lambda: Path(paths["snapshot_root"]),
    )
    effective = {
        "sound_tokenizer": {
            "bucket_name": "",
            "avae_path": str(
                Path(paths["sound_directory"]) / snapshot._SOUND_CHECKPOINT
            ),
        },
        "vlm_config": {"tokenizer": {"tokenizer_type": paths["snapshot_root"]}},
        "compile": {"enabled": False},
        "fixture_native_config": True,
    }
    return setup, SimpleNamespace(model_config=effective)


def test_resolve_reads_metadata_without_materialization(native_fixture, monkeypatch):
    monkeypatch.setattr(
        snapshot,
        "materialize_nano_snapshot",
        lambda _: pytest.fail("weights downloaded"),
    )
    assert snapshot.resolve_nano_revision() == _REVISION


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39, "A" * 40, "a" * 41])
def test_materialization_rejects_mutable_or_malformed_revision(revision, monkeypatch):
    monkeypatch.setattr(
        snapshot, "_materialized_paths", lambda _: pytest.fail("download attempted")
    )
    with pytest.raises(snapshot.NanoCheckpointError, match="full lowercase SHA"):
        snapshot.materialize_nano_snapshot(revision)


def test_materialization_binds_native_auxiliary_and_unique_selected_files(
    native_fixture,
):
    receipt = snapshot.materialize_nano_snapshot(_REVISION)
    binding = receipt["model_binding"]
    assert binding["revision"] == _REVISION
    assert binding["wan_vae"]["revision"] == _WAN_REVISION
    assert list(binding["files"]) == sorted(binding["files"])
    assert (
        len([name for name in binding["files"] if name.endswith("weights.safetensors")])
        == 1
    )
    assert binding["sound_conversion"]["state_dict_sha256"]
    assert str(native_fixture.nano) not in json.dumps(binding)
    assert snapshot._SOUND_CHECKPOINT not in binding["files"]
    assert receipt["derived_file_hashes"]["sound_checkpoint"]


def test_prepare_uses_native_config_without_enabling_training_or_changing_samples(
    native_fixture,
):
    samples = {"steps": 35, "window": 93, "normalize_cfg": False, "native_rgb": True}
    args = SimpleNamespace(
        checkpoint_path="Cosmos3-Nano",
        experiment_overrides=[],
        use_torch_compile=True,
        sample_overrides=copy.deepcopy(samples),
    )
    receipt = snapshot.prepare_native_setup(args, _REVISION)
    assert args.checkpoint_path == receipt["local_paths"]["snapshot_root"]
    assert args.config_file == receipt["local_paths"]["config_file"]
    assert args.experiment_overrides == [
        "model.config.sound_tokenizer.from_checkpoint=true"
    ]
    assert args.use_torch_compile is False
    assert args.sample_overrides == samples
    assert native_fixture.checkpoints[snapshot._SOUND_URI].hf.revision == "main"


def test_prepare_rejects_another_selected_checkpoint_before_download(monkeypatch):
    monkeypatch.setattr(
        snapshot,
        "materialize_nano_snapshot",
        lambda _: pytest.fail("download attempted"),
    )
    with pytest.raises(snapshot.NanoCheckpointError, match="named Nano checkpoint"):
        snapshot.prepare_native_setup(
            SimpleNamespace(checkpoint_path="Cosmos3-Super"), _REVISION
        )


def test_full_snapshot_binds_additional_processor_and_model_config_files(
    native_fixture,
):
    additional = [
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vision_encoder/config.json",
        "transformer/config.json",
    ]
    for name in additional:
        (native_fixture.nano / name).write_text("fixture-extra-configuration")
    native_fixture.metadata[snapshot._REPOSITORY] = _remote_metadata(
        native_fixture.nano, _REVISION
    )
    receipt = snapshot.materialize_nano_snapshot(_REVISION)
    assert {"nano/" + name for name in additional}.issubset(
        receipt["model_binding"]["files"]
    )


@pytest.mark.parametrize(
    "name",
    ["preprocessor_config.json", "tokenizer.json", "vision_encoder/model.safetensors"],
)
def test_missing_selected_files_fail_closed(native_fixture, name):
    (native_fixture.nano / name).unlink()
    with pytest.raises((snapshot.NanoCheckpointError, FileNotFoundError)):
        snapshot.materialize_nano_snapshot(_REVISION)


@pytest.mark.parametrize(
    "unsafe", ["../private-file", "/private-file", "a/../b", "a\\b", "a//b"]
)
def test_index_traversal_is_rejected_before_selected_file_reads(native_fixture, unsafe):
    index = native_fixture.nano / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"fixture": unsafe}}))
    with pytest.raises(snapshot.NanoCheckpointError, match="unsafe relative path"):
        snapshot.materialize_nano_snapshot(_REVISION)


@pytest.mark.parametrize(
    "filename", ["transformer/weights.safetensors", "preprocessor_config.json"]
)
def test_corrupt_selected_lfs_or_git_object_is_rejected(native_fixture, filename):
    path = native_fixture.nano / filename
    raw = path.read_bytes()
    path.write_bytes(b"x" + raw[1:])
    with pytest.raises(snapshot.NanoCheckpointError, match="immutable.*object"):
        snapshot.materialize_nano_snapshot(_REVISION)


def test_remote_revision_mismatch_is_rejected(native_fixture):
    native_fixture.metadata[snapshot._REPOSITORY].sha = "c" * 40
    with pytest.raises(
        snapshot.NanoCheckpointError, match="different immutable revision"
    ):
        snapshot.materialize_nano_snapshot(_REVISION)


def test_different_default_sound_converter_is_rejected(native_fixture):
    native_fixture.checkpoints[snapshot._SOUND_URI].post_download = lambda _: None
    with pytest.raises(
        snapshot.NanoCheckpointError, match="sound checkpoint contract changed"
    ):
        snapshot.materialize_nano_snapshot(_REVISION)


def test_existing_derived_sound_tensors_are_validated(native_fixture):
    snapshot.materialize_nano_snapshot(_REVISION)
    checkpoint = native_fixture.nano / "sound_tokenizer" / snapshot._SOUND_CHECKPOINT
    payload = json.loads(checkpoint.read_text())
    payload["decoder.layers.1.weight"]["values"][0] = 999
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(snapshot.NanoCheckpointError, match="changed a selected tensor"):
        snapshot.materialize_nano_snapshot(_REVISION)


def test_portable_binding_ignores_derived_serialization_layout(native_fixture):
    first = snapshot.materialize_nano_snapshot(_REVISION)
    checkpoint = native_fixture.nano / "sound_tokenizer" / snapshot._SOUND_CHECKPOINT
    checkpoint.write_text(json.dumps(json.loads(checkpoint.read_text()), indent=4))
    second = snapshot.materialize_nano_snapshot(_REVISION)
    assert first["model_binding"] == second["model_binding"]
    assert first["derived_file_hashes"] != second["derived_file_hashes"]


def test_portable_binding_and_effective_configuration_survive_worker_path_changes(
    native_fixture, tmp_path
):
    original = snapshot.materialize_nano_snapshot(_REVISION)
    source_root = native_fixture.nano.parents[2]
    replacement_root = tmp_path / "replacement-worker"
    shutil.copytree(source_root, replacement_root)
    relocated = copy.deepcopy(original)
    relocated["local_paths"] = {
        key: value.replace(str(source_root), str(replacement_root))
        for key, value in original["local_paths"].items()
    }
    native_fixture.checkpoints[snapshot._WAN_URI].path = Path(
        relocated["local_paths"]["wan_file"]
    )
    original_setup, original_pipe = _resolved_native(original)
    setup, pipe = _resolved_native(relocated)
    receipt = snapshot.record_native_model_selection(
        setup, pipe, relocated, receipt_path=tmp_path / "relocated.json"
    )
    original_effective = snapshot._effective_config(
        original_pipe, original["local_paths"]
    )
    assert receipt["model_binding"] == original["model_binding"]
    assert receipt["effective_config"] == original_effective


def test_file_changed_while_hashing_is_rejected(native_fixture, monkeypatch):
    path = native_fixture.nano / "transformer/weights.safetensors"
    actual = snapshot._file_identity(path)
    calls = iter([actual, (*actual[:3], actual[3] + 1)])
    monkeypatch.setattr(snapshot, "_file_identity", lambda _: next(calls))
    with pytest.raises(snapshot.NanoCheckpointError, match="changed during hashing"):
        snapshot._file_hash(path)


def test_selection_rereads_selected_bytes_and_records_actual_effective_config(
    native_fixture, tmp_path
):
    preparation = snapshot.materialize_nano_snapshot(_REVISION)
    setup, pipe = _resolved_native(preparation)
    destination = tmp_path / "model-selection.json"
    receipt = snapshot.record_native_model_selection(
        setup, pipe, preparation, receipt_path=destination
    )
    assert receipt["model_binding"] == preparation["model_binding"]
    assert receipt["effective_config"]["fixture_native_config"] is True
    assert (
        receipt["effective_config"]["vlm_config"]["tokenizer"]["tokenizer_type"]
        == "binding://snapshot_root"
    )
    assert receipt["native_runtime"]["loader_open_observed"] is False
    assert str(native_fixture.nano) not in destination.read_text()
    assert json.loads(destination.read_text()) == receipt
    assert destination.stat().st_mode & 0o777 == 0o600
    encoded = json.dumps(
        receipt["effective_config"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert receipt["effective_config_sha256"] == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("changed", ["source", "model", "wan"])
def test_selection_rejects_bytes_changed_after_preparation(
    native_fixture, tmp_path, changed
):
    preparation = snapshot.materialize_nano_snapshot(_REVISION)
    setup, pipe = _resolved_native(preparation)
    path = {
        "source": native_fixture.framework / snapshot._SOURCES[0],
        "model": native_fixture.nano / "transformer/weights.safetensors",
        "wan": native_fixture.wan,
    }[changed]
    path.write_text("changed-fixture-bytes")
    destination = tmp_path / "model-selection.json"
    with pytest.raises(snapshot.NanoCheckpointError):
        snapshot.record_native_model_selection(
            setup, pipe, preparation, receipt_path=destination
        )
    assert not destination.exists()


@pytest.mark.parametrize("changed", ["checkpoint", "processor", "sound", "compile"])
def test_selection_rejects_other_actual_native_selection(
    native_fixture, tmp_path, changed
):
    preparation = snapshot.materialize_nano_snapshot(_REVISION)
    setup, pipe = _resolved_native(preparation)
    if changed == "checkpoint":
        setup.download_checkpoint = lambda: tmp_path
    elif changed == "processor":
        pipe.model_config["vlm_config"]["tokenizer"]["tokenizer_type"] = "other-fixture"
    elif changed == "sound":
        pipe.model_config["sound_tokenizer"]["avae_path"] = str(
            tmp_path / "other-fixture.ckpt"
        )
    else:
        setup.use_torch_compile = True
    with pytest.raises(snapshot.NanoCheckpointError):
        snapshot.record_native_model_selection(
            setup, pipe, preparation, receipt_path=tmp_path / "selection.json"
        )


def test_internal_cli_writes_only_exact_receipt_path(native_fixture, tmp_path, capsys):
    destination = tmp_path / "resolved-revision.json"
    assert snapshot.main(["--mode", "resolve", "--receipt-path", str(destination)]) == 0
    assert json.loads(destination.read_text())["revision"] == _REVISION
    assert capsys.readouterr().out == ""


def test_internal_cli_validation_failure_delivers_safe_failure_receipt(
    tmp_path, capsys
):
    destination = tmp_path / "validation-failure.json"
    assert (
        snapshot.main(
            [
                "--mode",
                "materialize",
                "--revision",
                "mutable-main",
                "--receipt-path",
                str(destination),
            ]
        )
        == 1
    )
    assert json.loads(destination.read_text()) == {
        "schema": "npa.cosmos3.nano-failure.v1",
        "reason": "Nano checkpoint revision must be a full lowercase SHA",
    }
    assert capsys.readouterr().out == ""
    assert destination.stat().st_mode & 0o777 == 0o600
