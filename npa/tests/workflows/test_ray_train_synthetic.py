"""Native Train reference contracts; synthetic fixtures never claim GPU execution."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


EXAMPLE = Path(__file__).parents[2] / "workflows/workbench/ray-train-synthetic"


@pytest.fixture(autouse=True)
def isolate_example_imports(monkeypatch):
    """Keep the standalone source imports separate from other application examples."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    names = ("train", "artifacts", "inspect_results")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    yield
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def load(name):
    """Load a standalone application file without importing Ray or Torch."""
    spec = importlib.util.spec_from_file_location(f"ray_train_example_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recipe():
    """Supply a tiny explicit protocol fixture, unrelated to real CUDA evidence."""
    return dict(workers=2, steps=4, samples_per_rank=8, checkpoint_interval=2,
                learning_rate=0.1, seed=7, fail_after_step=0)


@pytest.fixture
def journal(recipe):
    """Supply finite deterministic rows for validation and actual RRD roundtrips."""
    return [dict(optimizer_step=step, loss=1 / step, gradient_norm=0.5, parameter_delta=0.1,
                 learning_rate=0.1, samples_per_second=10., ranks=[dict(
                     rank=rank, world_size=2, device_type="cuda", device_fingerprint=str(rank) * 64,
                     parameter_sha256="a" * 64, restored_from_step=0,
                 ) for rank in range(2)]) for step in range(1, 5)]


@pytest.mark.parametrize("key,value", [("workers", 1), ("steps", 0), ("steps", 1), ("samples_per_rank", -1),
                                      ("checkpoint_interval", 0), ("learning_rate", float("nan")),
                                      ("learning_rate", 1.1), ("fail_after_step", 3), ("fail_after_step", 4)])
def test_invalid_recipe_fails_before_ray(key, value, recipe):
    recipe[key] = value
    with pytest.raises(ValueError):
        load("train").validate_recipe(recipe)


@pytest.mark.parametrize("mutation", ["missing_step", "duplicate_step", "nan_loss", "zero_update",
                                      "cpu_rank", "wrong_world", "missing_rank", "same_device",
                                      "divergent_model", "no_learning", "wrong_lr"])
def test_bad_progress_or_rank_evidence_cannot_be_exported(mutation, journal, recipe):
    if mutation == "missing_step":
        journal.pop()
    elif mutation == "duplicate_step":
        journal[1]["optimizer_step"] = 1
    elif mutation == "nan_loss":
        journal[1]["loss"] = float("nan")
    elif mutation == "zero_update":
        journal[1]["parameter_delta"] = 0
    elif mutation == "cpu_rank":
        journal[1]["ranks"][1]["device_type"] = "cpu"
    elif mutation == "wrong_world":
        journal[1]["ranks"][1]["world_size"] = 1
    elif mutation == "missing_rank":
        journal[1]["ranks"].pop()
    elif mutation == "same_device":
        journal[1]["ranks"][1]["device_fingerprint"] = "0" * 64
    elif mutation == "divergent_model":
        journal[1]["ranks"][1]["parameter_sha256"] = "b" * 64
    elif mutation == "no_learning":
        journal[-1]["loss"] = journal[0]["loss"]
    else:
        journal[1]["learning_rate"] = 0.5
    with pytest.raises(ValueError):
        load("train").validate_journal(journal, recipe)


def test_every_restarted_rank_must_restore_exact_committed_step(journal, recipe):
    recipe["fail_after_step"] = 2
    for row in journal[2:]:
        for rank in row["ranks"]:
            rank["restored_from_step"] = 2
    load("train").validate_journal(journal, recipe)
    journal[2]["ranks"][1]["restored_from_step"] = 0
    with pytest.raises(ValueError, match="Every worker"):
        load("train").validate_journal(journal, recipe)


@pytest.fixture
def exported(tmp_path, journal, recipe):
    """Write real RRD from explicit fixture values, with a clearly fake checkpoint."""
    tmp_path = tmp_path / "export"
    tmp_path.mkdir()
    train = load("train")
    report = {"run_name": "synthetic-test", "recipe": recipe}
    (tmp_path / "metrics.json").write_text(json.dumps(journal))
    (tmp_path / "state.pt").write_bytes(b"explicit non-Torch checksum fixture")
    report.update(checkpoint_sha256=train.digest(tmp_path / "state.pt"),
                  journal_sha256=train.digest(tmp_path / "metrics.json"))
    train.write_recording(tmp_path / "metrics.rrd", journal, report)
    report["rrd_sha256"] = train.digest(tmp_path / "metrics.rrd")
    (tmp_path / "result.json").write_text(json.dumps(report))
    write_hashes(tmp_path)
    return tmp_path


def write_hashes(directory):
    """Bind test export bytes independently of the RRD contents."""
    train = load("train")
    (directory / "SHA256SUMS").write_text("".join(
        f"{train.digest(path)}  {path.name}\n" for path in sorted(directory.iterdir())
        if path.name != "SHA256SUMS"
    ))


def test_complete_rrd_timeline_roundtrip(exported):
    assert load("inspect_results").inspect(exported) == dict(
        artifacts_verified=4, optimizer_steps_decoded=4, metric_entities_decoded=5,
        checkpoint_events_decoded=2,
    )


def test_checksum_corruption_is_rejected(exported):
    (exported / "state.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        load("inspect_results").inspect(exported)


def test_rehashed_journal_cannot_hide_rrd_value_drift(exported, journal):
    changed = copy.deepcopy(journal)
    changed[1]["loss"] = 99.
    report = json.loads((exported / "result.json").read_text())
    report.pop("rrd_sha256")
    load("train").write_recording(exported / "metrics.rrd", changed, report)
    report["rrd_sha256"] = load("train").digest(exported / "metrics.rrd")
    (exported / "result.json").write_text(json.dumps(report))
    write_hashes(exported)
    with pytest.raises(ValueError, match="values differ"):
        load("inspect_results").inspect(exported)


def test_rrd_missing_step_cannot_pass_last_step_only_check(exported, journal, recipe):
    report = json.loads((exported / "result.json").read_text())
    report.pop("rrd_sha256")
    load("train").write_recording(exported / "metrics.rrd", [journal[0], *journal[2:]],
                                 report)
    report["rrd_sha256"] = load("train").digest(exported / "metrics.rrd")
    (exported / "result.json").write_text(json.dumps(report))
    write_hashes(exported)
    with pytest.raises(ValueError, match="timeline differs"):
        load("inspect_results").inspect(exported)


def test_manifest_cannot_escape_export_directory(exported):
    (exported / "SHA256SUMS").write_text("a" * 64 + "  ../private\n")
    with pytest.raises(ValueError, match="exactly"):
        load("inspect_results").inspect(exported)


def test_cli_rejects_local_storage_before_ray(tmp_path):
    result = subprocess.run([sys.executable, str(EXAMPLE / "train.py"), "--storage-path", "/tmp/checkpoints",
                             "--output-dir", str(tmp_path / "out"), "--run-name", "test"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "run-scoped unsigned s3://" in result.stderr
    assert not (tmp_path / "out").exists()


def test_cluster_is_guarded_native_train_resource_profile():
    profile = yaml.safe_load((EXAMPLE / "cluster.yaml").read_text())
    assert profile["num_nodes"] == 2
    assert profile["resources"]["accelerators"] == "B200:1"
    assert "@sha256:" in profile["resources"]["image_id"]
    assert profile["run"].strip() == "bash /tmp/ray-train-bootstrap/start.sh"
    assert "TorchTrainer(" in (EXAMPLE / "train.py").read_text()
    assert "JobSubmissionClient" not in (EXAMPLE / "train.py").read_text()
    assert "ray stop" not in (EXAMPLE / "cluster/start.sh").read_text()


@pytest.mark.parametrize("uri", ["/tmp/local", "s3://bucket", "s3://bucket/a/../b",
                                 "s3://bucket/a?signature=secret", "s3://user:pass@bucket/a"])
def test_storage_rejects_ambiguous_or_unsigned_destination(uri):
    with pytest.raises(ValueError, match="unsigned"):
        load("artifacts").storage(uri)


def test_storage_requires_complete_process_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://storage.example.test")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic")
    with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY"):
        load("artifacts").storage("s3://bucket/train")


def test_remote_exports_are_read_back_and_downloaded_exactly(exported, tmp_path):
    from pyarrow import fs

    remote = tmp_path / "remote"
    (remote / "exports").mkdir(parents=True)
    filesystem = fs.SubTreeFileSystem(str(remote), fs.LocalFileSystem())
    artifacts = load("artifacts")
    artifacts.publish(filesystem, "exports", exported)
    downloaded = tmp_path / "downloaded"
    artifacts.download(filesystem, "exports", downloaded)
    assert load("inspect_results").inspect(downloaded)["optimizer_steps_decoded"] == 4
    for path in exported.iterdir():
        assert path.read_bytes() == (downloaded / path.name).read_bytes()


def test_existing_remote_export_is_never_overwritten(exported, tmp_path):
    from pyarrow import fs

    remote = tmp_path / "remote"
    (remote / "exports").mkdir(parents=True)
    filesystem = fs.SubTreeFileSystem(str(remote), fs.LocalFileSystem())
    (remote / "exports/metrics.json").write_bytes(b"prior evidence")
    with pytest.raises(ValueError, match="Existing export differs"):
        load("artifacts").publish(filesystem, "exports", exported)
    assert (remote / "exports/metrics.json").read_bytes() == b"prior evidence"


def test_retry_partial_upload_preserves_exact_rrd_bytes(exported, tmp_path):
    from pyarrow import fs

    remote = tmp_path / "remote"
    (remote / "exports").mkdir(parents=True)
    filesystem = fs.SubTreeFileSystem(str(remote), fs.LocalFileSystem())

    class FailReadbackOnce:
        """Inject a real transfer boundary failure after the RRD write completed."""

        def __getattr__(self, name):
            return getattr(filesystem, name)

        def open_input_file(self, path):
            if path.endswith("metrics.rrd"):
                raise OSError("injected readback failure")
            return filesystem.open_input_file(path)

    artifacts = load("artifacts")
    with pytest.raises(OSError, match="readback"):
        artifacts.publish(FailReadbackOnce(), "exports", exported)
    rrd_bytes = (remote / "exports/metrics.rrd").read_bytes()
    assert not (remote / "exports/SHA256SUMS").exists()
    artifacts.publish(filesystem, "exports", exported)
    assert (remote / "exports/metrics.rrd").read_bytes() == rrd_bytes
    artifacts.download(filesystem, "exports", tmp_path / "retried")
    assert load("inspect_results").inspect(tmp_path / "retried")["optimizer_steps_decoded"] == 4


def test_unlisted_private_file_blocks_all_uploads(exported, tmp_path):
    from pyarrow import fs

    (exported / "private.env").write_text("unrelated private fixture")
    remote = tmp_path / "remote"
    remote.mkdir()
    with pytest.raises(ValueError, match="unexpected"):
        load("artifacts").publish(fs.LocalFileSystem(), str(remote), exported)
    assert list(remote.iterdir()) == []
    with pytest.raises(ValueError, match="unexpected"):
        load("inspect_results").inspect(exported)


def test_rehashed_report_cannot_hide_stale_rrd_provenance(exported):
    report = json.loads((exported / "result.json").read_text())
    report["recipe"]["seed"] = 99
    (exported / "result.json").write_text(json.dumps(report))
    write_hashes(exported)
    with pytest.raises(ValueError, match="provenance differs"):
        load("inspect_results").inspect(exported)


def test_checksum_manifest_cannot_follow_external_symlink(exported, tmp_path):
    manifest = exported / "SHA256SUMS"
    external = tmp_path / "external-manifest"
    manifest.rename(external)
    manifest.symlink_to(external)
    with pytest.raises(ValueError, match="regular"):
        load("inspect_results").inspect(exported)


@pytest.mark.parametrize("first_fails", [False, True])
def test_live_cleanup_reconciles_lost_response_and_attempts_every_owned_job(tmp_path, first_fails):
    """A lost submission response or one cancel error must not orphan later Jobs."""
    path = Path(__file__).parents[1] / "e2e/test_ray_train_synthetic_live.py"
    spec = importlib.util.spec_from_file_location("train_live_cleanup_contract", path)
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    active = {"accepted-response-lost", "second-owned", "someone-else"}
    stopped = []

    class Client:
        def list_jobs(self):
            return [SimpleNamespace(submission_id=name) for name in active | set(stopped)]

        def get_job_status(self, job):
            return SimpleNamespace(is_terminal=lambda: job not in active)

        def stop_job(self, job):
            if first_fails and job == "accepted-response-lost":
                raise ConnectionError("injected cancel transport error")
            active.remove(job)
            stopped.append(job)

        def get_job_logs(self, job):
            return f"Preserved {job}"

    errors = live._cancel_owned_jobs(Client(), ["accepted-response-lost", "second-owned"], tmp_path)
    assert "second-owned" in stopped
    assert "someone-else" in active
    assert (tmp_path / "second-owned.log").stat().st_mode & 0o077 == 0
    if first_fails:
        assert errors == [{"job": "accepted-response-lost", "phase": "cancel", "error": "ConnectionError"}]
    else:
        assert not errors and "accepted-response-lost" in stopped
