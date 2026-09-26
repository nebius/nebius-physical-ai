"""Exercise checkpoint retention across real publication and cleanup boundaries."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.flex_pi import training
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training_cleanup import cleanup_checkpoint_copies


STATE_FILES = (
    "model.safetensors",
    "optimizer.bin",
    "scheduler.bin",
    "trainer_state.json",
    "dataset_stats.json",
    *(f"random_states_{rank}.pkl" for rank in range(4)),
)


class MemoryStorage:
    def __init__(self, case):
        self.case = case
        self.objects = {}

    def upload_file(self, path, uri):
        if self.case.failure == "upload:" + uri.rsplit("/", 1)[-1]:
            raise OSError("publication unavailable")
        self.objects[uri] = Path(path).read_bytes()

    def download_file(self, uri, path):
        payload = self.objects[uri]
        if self.case.failure == "readback:" + uri.rsplit("/", 1)[-1]:
            payload = b"corrupt readback"
        Path(path).write_bytes(payload)
        if uri.endswith("/result.json"):
            self.case.after_result_readback()


def _state(directory):
    directory.mkdir(parents=True)
    for name in STATE_FILES:
        (directory / name).write_bytes(b"checkpoint " + name.encode())


def _worker(case, plan, root):
    if plan["execution"]["mode"] == "resume":
        result = copy.deepcopy(case.resume)
        if case.failure == "resume":
            result["resume_probe"] = {"different": True}
        return result
    case.work = Path(plan["work_directory"])
    case.original = case.work / "checkpoints/state/step_001205"
    case.restored = case.work / "restored-state"
    _state(case.original)
    (case.work / "qualification.json").write_text("preserve diagnostic")
    weights = case.work / "checkpoints/weights"
    weights.mkdir()
    (weights / "step_001205.pt").write_bytes(b"unpublished weights")
    if case.failure == "worker":
        raise FlexPiError("worker failed")
    checkpoint = {
        "state_path": str(case.original),
        "step": 1205,
        "model_sha256": "model",
        "training_state": [{"rank": 0}],
    }
    case.resume = {
        "workload_sha256": "workload",
        "normalization_sha256": "stats",
        "loaded_model_sha256": "model",
        "loaded_training_state": [{"rank": 0}],
        "loaded_step": 1205,
        "resume_probe": {"next_step": 1206},
    }
    return {
        "checkpoint": checkpoint,
        "workload_sha256": "workload",
        "normalization_sha256": "stats",
        "resume_probe": {"next_step": 1206},
        "reference_benchmark_beaten": False,
    }


@pytest.fixture
def case(tmp_path, monkeypatch):
    case = SimpleNamespace(failure="", after_result_readback=lambda: None)
    case.storage = MemoryStorage(case)
    case.cache = tmp_path / "cache"
    case.request = training.TrainingRequest(output_path="s3://example-bucket/run")
    monkeypatch.setenv("NPA_MODEL_CACHE_DIR", str(case.cache))
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment", lambda: case.storage
    )
    monkeypatch.setattr(
        training, "_run_phase", lambda plan, root: _worker(case, plan, root)
    )
    return case


def test_success_removes_only_published_copies_after_result_readback(case):
    assets = case.cache / "flex-pi-training/assets"
    sibling = case.cache / "flex-pi-training/run-sibling"
    for directory in (assets, sibling):
        directory.mkdir(parents=True)
        (directory / "keep.bin").write_bytes(b"unrelated")
    case.after_result_readback = lambda: (case.original / "keep.txt").write_text(
        "new diagnostic"
    )
    result = training.run_training(case.request)
    assert result["checkpoint_resume_verified"] is True
    assert (
        json.loads(case.storage.objects[case.request.output_path + "/result.json"])
        == result
    )
    assert "state_path" not in result["checkpoint"]
    for name in STATE_FILES:
        assert not (case.original / name).exists()
    assert not case.restored.exists()
    assert (case.original / "keep.txt").read_text() == "new diagnostic"
    assert (case.work / "qualification.json").read_text() == "preserve diagnostic"
    assert (
        case.work / "checkpoints/weights/step_001205.pt"
    ).read_bytes() == b"unpublished weights"
    assert all(
        (directory / "keep.bin").read_bytes() == b"unrelated"
        for directory in (assets, sibling)
    )


@pytest.mark.parametrize(
    "failure",
    [
        "upload:manifest.json",
        "readback:manifest.json",
        "upload:result.json",
        "readback:result.json",
        "upload:optimizer.bin",
        "readback:optimizer.bin",
        "resume",
        "qualification",
        "worker",
    ],
)
def test_any_failure_before_verified_publication_preserves_checkpoint(
    case, monkeypatch, failure
):
    case.failure = failure
    if failure == "qualification":

        def qualify(*args, **kwargs):
            if kwargs.get("checkpoint") is not None:
                raise FlexPiError("qualification rejected")

        monkeypatch.setattr(training, "_qualify_memory_fill", qualify)
    with pytest.raises((OSError, FlexPiError)):
        training.run_training(case.request)
    assert all(
        (case.original / name).read_bytes() == b"checkpoint " + name.encode()
        for name in STATE_FILES
    )
    if failure in {
        "upload:manifest.json",
        "readback:manifest.json",
        "upload:result.json",
        "readback:result.json",
        "resume",
        "qualification",
    }:
        assert all(
            (case.restored / name).read_bytes() == b"checkpoint " + name.encode()
            for name in STATE_FILES
        )


def test_changed_restored_copy_preserves_both_sets_before_any_deletion(case, capsys):
    case.after_result_readback = lambda: (case.restored / "optimizer.bin").write_bytes(
        b"x" * len(b"checkpoint optimizer.bin")
    )
    result = training.run_training(case.request)
    assert result["checkpoint_resume_verified"] is True
    assert all((case.original / name).exists() for name in STATE_FILES)
    assert all((case.restored / name).exists() for name in STATE_FILES)
    assert "cleanup incomplete" in capsys.readouterr().err


def test_cleanup_io_failure_does_not_fail_or_contaminate_published_result(
    case, monkeypatch, capsys
):
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.name == "optimizer.bin":
            raise OSError("private failure s3://private-location/secret")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    result = training.run_training(case.request)
    assert (
        json.loads(case.storage.objects[case.request.output_path + "/result.json"])
        == result
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == "Published checkpoint cache cleanup incomplete; retained files require operator review.\n"
    )
    assert (case.original / "optimizer.bin").exists()
    assert (case.restored / "optimizer.bin").exists()


@pytest.fixture
def copies(tmp_path):
    work = tmp_path / "run-owned"
    original = work / "checkpoints/state/step_000030"
    restored = work / "restored-state"
    for directory in (original, restored):
        directory.mkdir(parents=True)
        (directory / "state.bin").write_bytes(b"state")
    manifest = {
        "files": [
            {
                "path": "state.bin",
                "size": 5,
                "sha256": hashlib.sha256(b"state").hexdigest(),
            }
        ]
    }
    return work, {
        "original": original,
        "restored": restored,
        "step": 30,
        "manifest": manifest,
    }


@pytest.mark.parametrize(
    "target", ["outside", "sibling", "traversal", "run-root", "wrong-step"]
)
def test_unowned_checkpoint_root_never_deletes_files(copies, tmp_path, target, capsys):
    work, arguments = copies
    original = arguments["original"]
    invalid = {
        "outside": tmp_path / "outside",
        "sibling": work.parent / "run-sibling",
        "traversal": work / "../run-sibling",
        "run-root": work,
        "wrong-step": work / "checkpoints/state/step_000031",
    }[target]
    if target != "traversal":
        invalid.mkdir(parents=True, exist_ok=True)
        (invalid / "sentinel").write_bytes(b"keep")
    arguments["original"] = invalid
    cleanup_checkpoint_copies(work, **arguments)
    assert (original / "state.bin").read_bytes() == b"state"
    assert (arguments["restored"] / "state.bin").read_bytes() == b"state"
    if target != "traversal":
        assert (invalid / "sentinel").read_bytes() == b"keep"
    assert "cleanup incomplete" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["root", "ancestor", "file"])
def test_symlink_in_either_copy_preserves_both_sets(copies, tmp_path, target):
    work, arguments = copies
    selected = {
        "root": arguments["restored"],
        "ancestor": arguments["original"].parent,
        "file": arguments["restored"] / "state.bin",
    }[target]
    moved = tmp_path / "preserved"
    selected.rename(moved)
    selected.symlink_to(moved, target_is_directory=target != "file")
    cleanup_checkpoint_copies(work, **arguments)
    assert all(
        (arguments[name] / "state.bin").read_bytes() == b"state"
        for name in ("original", "restored")
    )
    assert selected.is_symlink()


@pytest.mark.parametrize(
    "path", ["../outside", "/outside", "nested/../state.bin", "", ".", "./state.bin"]
)
def test_escaping_or_noncanonical_manifest_path_preserves_both_copies(copies, path):
    work, arguments = copies
    arguments["manifest"]["files"][0]["path"] = path
    cleanup_checkpoint_copies(work, **arguments)
    assert all(
        (arguments[name] / "state.bin").exists() for name in ("original", "restored")
    )


def test_nested_manifest_files_removed_but_unlisted_files_and_symlinks_retained(
    copies, tmp_path
):
    work, arguments = copies
    for name in ("original", "restored"):
        directory = arguments[name]
        (directory / "nested").mkdir()
        (directory / "state.bin").rename(directory / "nested/state.bin")
        (directory / "unexpected").symlink_to(tmp_path / "missing")
    arguments["manifest"]["files"][0]["path"] = "nested/state.bin"
    cleanup_checkpoint_copies(work, **arguments)
    for name in ("original", "restored"):
        assert not (arguments[name] / "nested").exists()
        assert (arguments[name] / "unexpected").is_symlink()


@pytest.mark.parametrize("dry_run", [False, True])
def test_profile_and_dry_run_never_cleanup_checkpoint_copies(
    case, monkeypatch, dry_run
):
    monkeypatch.setattr(
        "npa.workbench.flex_pi.training_cleanup.cleanup_checkpoint_copies",
        lambda *args, **kwargs: pytest.fail("unexpected cleanup"),
    )
    request = training.TrainingRequest(
        output_path=case.request.output_path, mode="profile", dry_run=dry_run
    )
    training.run_training(request)
    if dry_run:
        assert not case.cache.exists()
    else:
        assert all((case.original / name).exists() for name in STATE_FILES)
