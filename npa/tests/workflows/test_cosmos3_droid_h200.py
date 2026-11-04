"""Verify DROID range-reader semantics, failure handling and native launch settings."""

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-droid-h200"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"droid_{name}", RECIPE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fsspec(monkeypatch):
    module = SimpleNamespace(open=lambda path, mode="rb": open(path, mode))
    monkeypatch.setitem(sys.modules, "fsspec", module)
    return module


@pytest.fixture
def reader(tmp_path, fsspec):
    module = _load("native_video")
    manifest = tmp_path / "manifest.json"
    data = bytes(range(251)) * 9000
    key = "success/videos/camera/chunk-000/file-000.mp4"
    manifest.write_text(json.dumps({"files": [{"path": key, "size": len(data)}]}))
    settings = {
        "dataset_root": str(tmp_path / "dataset"),
        "manifest": str(manifest),
        "endpoint_url": "https://storage.example.test",
        "region": "test-region",
        "bucket": "test-bucket",
    }
    module.install(settings)
    calls = []

    def get_object(**kwargs):
        calls.append(kwargs)
        first, last = map(int, kwargs["Range"].removeprefix("bytes=").split("-"))
        return {
            "Body": io.BytesIO(data[first : last + 1]),
            "ResponseMetadata": {"HTTPStatusCode": 206},
            "ContentRange": f"bytes {first}-{last}/{len(data)}",
        }

    module._client = SimpleNamespace(get_object=get_object)
    module._client_process = module.os.getpid()
    yield module, settings, key, data, calls
    module.uninstall()


def test_reader_matches_local_bytes_across_seek_buffer_and_eof(reader, fsspec):
    module, settings, key, data, calls = reader
    path = Path(settings["dataset_root"]) / key
    with fsspec.open(str(path), mode="rb") as stream:
        assert stream.read(11) == data[:11]
        assert stream.seek(5) == 5
        assert stream.read(8) == data[5:13]
        assert len(calls) == 1
        stream.seek(1024 * 1024 - 5)
        assert stream.read(20) == data[1024 * 1024 - 5 : 1024 * 1024 + 15]
        stream.seek(-13, io.SEEK_END)
        assert stream.read() == data[-13:]
        stream.seek(len(data) + 50)
        assert stream.read() == b""
        with pytest.raises(ValueError):
            stream.seek(-1)
    with pytest.raises(ValueError):
        stream.read(1)
    assert module.counters()["get_attempts"] == len(calls)


def test_only_exact_manifest_video_prefix_is_routed(reader, tmp_path, fsspec):
    _, settings, _, _, calls = reader
    local = tmp_path / "ordinary.txt"
    local.write_text("local metadata")
    with fsspec.open(local, mode="rt") as stream:
        assert stream.read() == "local metadata"
    for suffix in ["missing.mp4", "../escape.mp4"]:
        with pytest.raises(ValueError):
            fsspec.open(str(Path(settings["dataset_root"]) / "success/videos" / suffix))
    with pytest.raises(ValueError):
        fsspec.open(
            str(Path(settings["dataset_root"]) / "success/videos/a.mp4"), mode="wb"
        )
    assert calls == []


@pytest.mark.parametrize("failure", ["short", "status", "range", "exception"])
def test_bad_range_response_stops_without_substituting_samples(reader, failure):
    module, _, key, _, _ = reader
    body = io.BytesIO(b"bad")

    def get_object(**kwargs):
        if failure == "exception":
            raise RuntimeError("must not expose transport details")
        size = module._sizes[key]
        return {
            "Body": body,
            "ResponseMetadata": {"HTTPStatusCode": 200 if failure == "status" else 206},
            "ContentRange": "wrong"
            if failure == "range"
            else f"bytes 0-1048575/{size}",
        }

    module._client = SimpleNamespace(get_object=get_object)
    with pytest.raises(OSError, match="no sample substitution") as error:
        module._RangeFile(key).read(1)
    assert "transport details" not in str(error.value)
    assert module.counters()["get_errors"] == 1
    if failure != "exception":
        assert body.closed


def test_manifest_and_endpoint_validation_leave_opener_unchanged(tmp_path, fsspec):
    module = _load("native_video")
    original = fsspec.open
    settings = {"endpoint_url": "https://user:password@storage.example.test"}
    with pytest.raises(ValueError):
        module.install(settings)
    assert fsspec.open is original
    manifest = tmp_path / "manifest.json"
    for entries in [
        [{"path": "../escape", "size": 1}],
        [{"path": "success/videos/a", "size": True}],
    ]:
        manifest.write_text(json.dumps({"files": entries}))
        with pytest.raises(ValueError):
            module._object_sizes(manifest)


def test_native_command_preserves_batch_and_eager_controls(tmp_path):
    recipe = _load("recipe")
    settings = {
        "framework": str(tmp_path),
        "tokenizer": "/inputs/tokenizer",
        "filter": "/inputs/filter.json",
    }
    command = recipe._argv(settings, tmp_path, "run")
    assert "--nproc_per_node=8" in command
    assert "--max_restarts=0" in command
    assert "dataloader_train.dataloader.num_workers=4" in command
    assert "model.config.compile.enabled=False" in command
    assert "trainer.callbacks.compile_tokenizer.enabled=False" in command
    assert "--dryrun" in recipe._argv(settings, tmp_path, "dryrun")


def test_toml_rejects_unknown_upstream_instead_of_partial_replacement(tmp_path):
    recipe = _load("recipe")
    target = tmp_path / "examples/toml/sft_config/action_policy_droid_nano.toml"
    target.parent.mkdir(parents=True)
    target.write_text("[trainer]\nmax_iter = 10000\n")
    with pytest.raises(ValueError, match="differs from the measured recipe"):
        recipe._toml({"framework": str(tmp_path), "updates": 5})


def _prepared_recipe_inputs(tmp_path):
    paths = {
        name: tmp_path / name
        for name in [
            "framework",
            "manifest",
            "base_dcp",
            "vae",
            "tokenizer",
            "avae_cache",
        ]
    }
    paths["dataset_root"] = tmp_path / "droid_plus_lerobot_640x360_20260412"
    paths["framework"].mkdir()
    files = [
        paths["dataset_root"] / "success/meta/info.json",
        paths["base_dcp"] / "model/.metadata",
        paths["vae"],
        paths["tokenizer"] / "tokenizer.json",
    ] + [
        paths["avae_cache"] / name
        for name in [
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "avae_48k_noncausal_25hz_64ch.ckpt",
            "avae_48k_noncausal_25hz_64ch.json",
        ]
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    paths["manifest"].write_text(
        json.dumps({"files": [{"path": "success/videos/camera/video.mp4", "size": 1}]})
    )
    return SimpleNamespace(
        **paths,
        operation="plan",
        output_dir=tmp_path / "output",
        bucket="test-bucket",
        endpoint_url="https://storage.example.test",
        region="test-region",
        aws_profile=None,
        updates=1,
    )


@pytest.mark.parametrize("root_name", ["Cosmos3-DROID", "cosmos3-droid"])
@pytest.mark.parametrize("symlink", [False, True])
def test_incompatible_dataset_root_stops_before_launch_or_outputs(
    tmp_path, monkeypatch, root_name, symlink
):
    recipe = _load("recipe")
    args = _prepared_recipe_inputs(tmp_path)
    native_root = args.dataset_root
    generic_root = tmp_path / root_name
    native_root.rename(generic_root)
    if symlink:
        native_root.symlink_to(generic_root, target_is_directory=True)
    else:
        args.dataset_root = generic_root

    def reject_subprocess(*args, **kwargs):
        pytest.fail("incompatible dataset roots must fail before any subprocess")

    monkeypatch.setattr(recipe.subprocess, "check_output", reject_subprocess)
    monkeypatch.setattr(recipe.subprocess, "run", reject_subprocess)
    previous_umask = recipe.os.umask(0o077)
    try:
        with pytest.raises(
            ValueError, match="directory named droid_plus_lerobot_640x360_20260412"
        ):
            recipe._execute(args)
    finally:
        recipe.os.umask(previous_umask)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_native_dataset_root_preserves_resolved_training_path(
    tmp_path, monkeypatch, fsspec, symlink
):
    recipe = _load("recipe")
    args = _prepared_recipe_inputs(tmp_path)
    native_root = args.dataset_root
    if symlink:
        args.dataset_root = tmp_path / "dataset-alias"
        args.dataset_root.symlink_to(native_root, target_is_directory=True)
    monkeypatch.setitem(sys.modules, "native_video", _load("native_video"))
    monkeypatch.setattr(
        recipe.subprocess, "check_output", lambda *args, **kwargs: recipe._FRAMEWORK
    )
    commands = []
    monkeypatch.setattr(
        recipe.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    settings = recipe._settings(args)
    environment = recipe._environment(settings, args.output_dir)
    assert environment["DROID_ROOT"] == str(native_root.resolve())
    assert settings["dataset_root"] == environment["DROID_ROOT"]
    assert commands == [["git", "diff", "--quiet", "HEAD"]]


def test_native_dataset_root_still_requires_success_metadata(tmp_path):
    recipe = _load("recipe")
    root = tmp_path / "droid_plus_lerobot_640x360_20260412"
    root.mkdir()
    with pytest.raises(ValueError, match="success/meta/info.json"):
        recipe._verify_dataset_root(root)


def _verification_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("NPA_DROID_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(get_device_name=lambda: "NVIDIA H200"),
            __version__="test",
        ),
    )
    monkeypatch.setitem(
        sys.modules, "cosmos_framework.utils.callback", SimpleNamespace(Callback=object)
    )
    return _load("verification")


def _native_trainer(monkeypatch):
    saved = []

    class Trainer:
        def __init__(self, config):
            self.config = config
            self.callbacks = SimpleNamespace(_callbacks=[])
            self.checkpointer = SimpleNamespace(
                save=lambda **kwargs: saved.append(kwargs)
            )

        def _optimizer_step(self):
            return "updated"

    monkeypatch.setitem(
        sys.modules,
        "cosmos_framework.trainer",
        SimpleNamespace(ImaginaireTrainer=Trainer),
    )
    return Trainer, saved


def test_checkpoint_suppression_and_realized_update_receipt(monkeypatch, tmp_path):
    verification = _verification_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(verification, "_native_configuration", lambda config: {})
    Trainer, saved = _native_trainer(monkeypatch)
    verification.install()
    config = SimpleNamespace(
        dataloader_train=SimpleNamespace(max_samples_per_batch=32),
        trainer=SimpleNamespace(grad_accum_iter=32, max_iter=1),
    )
    trainer = Trainer(config)
    receipt = trainer._droid_receipt
    receipt.on_train_start(None)
    batch = {"video": [None] * 32}
    loss = SimpleNamespace()
    loss.detach = lambda: loss
    loss.float = lambda: loss
    loss.item = lambda: 1.25
    for _ in range(32):
        receipt.on_training_step_batch_start(None, batch)
        receipt.on_training_step_batch_end(None, batch, None, loss)
    assert trainer._optimizer_step() == "updated"
    receipt.on_training_step_end(None, batch, None, loss)
    receipt.on_train_end(None)
    trainer.checkpointer.save(iteration=1)
    assert saved == []
    report = json.loads((tmp_path / "verification/rank-0.json").read_text())
    assert report["terminal"] and report["samples"] == 1024
    assert report["updates"] == report["optimizer_operations"] == 1


@pytest.mark.parametrize("status", [200, 206])
def test_range_reader_bounds_body_consumption_before_rejection(reader, status):
    module, _, key, data, _ = reader
    reads = []

    class Body(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert size > 0
            return super().read(size)

    body = Body(data)
    response = {
        "Body": body,
        "ResponseMetadata": {"HTTPStatusCode": status},
        "ContentRange": f"bytes 0-1048575/{len(data)}",
    }
    module._client = SimpleNamespace(get_object=lambda **kwargs: response)
    with pytest.raises(OSError):
        module._RangeFile(key).read(1)
    assert reads == ([] if status == 200 else [1048577])
    assert body.closed


def test_update_reconciliation_rejects_incomplete_rank(tmp_path):
    recipe = _load("recipe")
    directory = tmp_path / "verification"
    directory.mkdir()
    for rank in range(8):
        row = {
            "rank": rank,
            "terminal": True,
            "finite_losses": True,
            "microsteps": 32,
            "samples": 1024,
            "updates": 1,
            "optimizer_operations": 1 if rank else 0,
        }
        (directory / f"rank-{rank}.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="did not realize"):
        recipe._verify_completed_updates({"updates": 1}, tmp_path)
