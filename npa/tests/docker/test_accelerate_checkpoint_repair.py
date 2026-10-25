"""Reject hostile checkpoint indexes while preserving regular cached shards."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


COMMON = Path(__file__).resolve().parents[2] / "docker/workbench/common"


@pytest.fixture
def repair():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_repair", COMMON / "repair_accelerate_checkpoints.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint_files(repair, tmp_path):
    path = tmp_path / "checkpoint_guard.py"
    path.write_text("import os\nimport json\n\n" + repair.INDEX_GUARD)
    spec = importlib.util.spec_from_file_location("checkpoint_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._npa_checkpoint_files


def _index(folder, value):
    folder.mkdir(exist_ok=True)
    index = folder / "model.index.json"
    index.write_text(json.dumps(value))
    return index


@pytest.mark.parametrize("version", ["1.12.0", "1.13.0", "1.14.0", "1.15.0"])
def test_unknown_source_is_never_patched(repair, version):
    with pytest.raises(RuntimeError, match="differs from reviewed upstream"):
        repair.patched_source(version, b"# unreviewed source\n")


def test_unknown_version_is_not_assumed_safe(repair):
    with pytest.raises(RuntimeError, match="requires reviewed Accelerate"):
        repair.patched_source("1.16.0", b"# upstream changed\n")


@pytest.mark.parametrize("wrapped", [False, True])
def test_regular_nested_shards_are_sorted_and_deduplicated(
    tmp_path, checkpoint_files, wrapped
):
    nested = tmp_path / "nested"
    nested.mkdir()
    for name in ("a.bin", "b.bin"):
        (nested / name).write_bytes(b"test weight shard")
    mapping = {"a": "nested/b.bin", "b": "nested/a.bin", "c": "nested/b.bin"}
    index = _index(tmp_path, {"weight_map": mapping} if wrapped else mapping)
    assert checkpoint_files(index) == [str(nested / "a.bin"), str(nested / "b.bin")]


def test_hugging_face_snapshot_symlinks_remain_usable(tmp_path, checkpoint_files):
    blob = tmp_path / "blobs" / "hash-of-public-weights"
    blob.parent.mkdir()
    blob.write_bytes(b"regular cached shard")
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "part.safetensors").symlink_to(blob)
    index = _index(snapshot, {"weight_map": {"weight": "part.safetensors"}})
    assert checkpoint_files(index) == [str(snapshot / "part.safetensors")]


@pytest.mark.parametrize(
    "entry",
    [
        "../outside.bin",
        "../../outside.bin",
        "../checkpoint-other/part.bin",
        "",
        None,
        3,
        [],
        {},
    ],
)
def test_index_cannot_select_paths_outside_checkpoint(
    tmp_path, checkpoint_files, entry
):
    index = _index(tmp_path / "checkpoint", {"weight_map": {"weight": entry}})
    (tmp_path / "outside.bin").write_bytes(b"private neighboring data")
    with pytest.raises(ValueError, match="checkpoint shard"):
        checkpoint_files(index)


def test_absolute_shard_path_is_rejected_even_inside_root(tmp_path, checkpoint_files):
    shard = tmp_path / "part.bin"
    shard.write_bytes(b"regular weights")
    index = _index(tmp_path, {"weight_map": {"weight": str(shard)}})
    with pytest.raises(ValueError, match="relative file path"):
        checkpoint_files(index)


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        "not a map",
        {},
        {"weight_map": None},
        {"weight_map": []},
        {"weight_map": {}},
    ],
)
def test_malformed_index_is_rejected(tmp_path, checkpoint_files, mapping):
    with pytest.raises(ValueError, match="nonempty mapping"):
        checkpoint_files(_index(tmp_path, mapping))


@pytest.mark.parametrize("kind", ["directory", "fifo", "fifo-symlink"])
def test_nonregular_shards_are_rejected_before_open(tmp_path, checkpoint_files, kind):
    shard = tmp_path / "part.bin"
    if kind == "directory":
        shard.mkdir()
    else:
        pipe = tmp_path / "named-pipe"
        os.mkfifo(pipe)
        if kind == "fifo-symlink":
            shard.symlink_to(pipe)
        else:
            pipe.rename(shard)
    index = _index(tmp_path, {"weight_map": {"weight": "part.bin"}})
    with pytest.raises(ValueError, match="regular file"):
        checkpoint_files(index)


def test_index_fifo_is_rejected_with_nonblocking_descriptor(
    tmp_path, checkpoint_files, monkeypatch
):
    index = tmp_path / "model.index.json"
    os.mkfifo(index)
    actual_open = os.open
    descriptors = []

    def checked_open(path, flags):
        assert flags & os.O_NONBLOCK
        descriptor = actual_open(path, flags)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", checked_open)
    with pytest.raises(ValueError, match="index must be a regular file"):
        checkpoint_files(index)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_cosmos_serving_repairs_reused_and_fresh_runtimes():
    image = COMMON.parent / "cosmos3-serving"
    script = (image / "runtime_bootstrap.sh").read_text()
    call = (
        '"${VENV}/bin/python" /opt/npa-cosmos3-serving/repair_accelerate_checkpoints.py'
    )
    reuse, install = script.split('work="${ROOT}/.install-$$"', 1)
    assert reuse.index(call) < reuse.index('exec "$@"')
    assert install.index('--no-deps "${work}/source"') < install.index(call)
    assert install.index(call) < install.index('touch "${MARKER}"')
