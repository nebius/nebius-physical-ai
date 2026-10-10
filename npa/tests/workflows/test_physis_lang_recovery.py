"""Exercise publication failures and recovery without GPU inference or live storage."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.clients.storage import StorageClient, StoragePreconditionFailed
from npa.workflows import physis_lang as cli
from npa.workflows import physis_lang_artifacts as artifacts


class ObjectStore:
    def __init__(self):
        self.objects = {}
        self.fail_upload = ""
        self.fail_readback = False

    def put_bytes_conditional(self, payload, uri, **kwargs):
        assert kwargs == {"if_none_match": True}
        if self.fail_upload and uri.endswith(self.fail_upload):
            self.fail_upload = ""
            raise OSError("upload interrupted")
        if uri in self.objects:
            raise StoragePreconditionFailed("already exists")
        self.objects[uri] = payload
        return "etag"

    def read_bytes_with_etag(self, uri):
        return (self.objects[uri], "etag") if uri in self.objects else None

    def download_directory(self, uri, destination):
        if self.fail_readback and uri.rstrip("/") + "/video.mp4" in self.objects:
            self.fail_readback = False
            raise OSError("readback interrupted")
        prefix = uri.rstrip("/") + "/"
        for key, value in self.objects.items():
            if key.startswith(prefix):
                path = Path(destination) / key[len(prefix) :]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)


@pytest.fixture
def store(monkeypatch):
    result = ObjectStore()
    monkeypatch.setattr(StorageClient, "from_environment", lambda: result)
    return result


@pytest.fixture
def stage(monkeypatch, tmp_path):
    roots = []

    def allocate(**kwargs):
        root = tmp_path / f"attempt-{len(roots)}"
        root.mkdir()
        roots.append(root)
        return str(root)

    monkeypatch.setattr(cli, "tempfile", SimpleNamespace(mkdtemp=allocate))
    generated = []

    def generate(args, root, output):
        generated.append(output)
        (output / "video.mp4").write_bytes(b"original GPU output")

    monkeypatch.setattr(cli, "_run", generate)
    return roots, generated


def argv(destination="s3://example-bucket/run/seed-a/"):
    return [
        "generate",
        "--seed",
        "0",
        "--input-path",
        "prepared",
        "--output-path",
        destination,
    ]


def test_readback_failure_preserves_evidence_and_retry_skips_generation(store, stage):
    store.fail_readback = True
    with pytest.raises(OSError, match="readback interrupted"):
        cli.main(argv())
    failures = [key for key in store.objects if key.endswith("failure.json")]
    assert len(failures) == 1
    assert json.loads(store.objects[failures[0]])["error_type"] == "OSError"
    assert any(
        "-failed/" in key and key.endswith("output/video.mp4") for key in store.objects
    )
    cli.main(argv())
    roots, generated = stage
    assert len(generated) == 1
    assert not any(root.exists() for root in roots)


def test_partial_upload_can_be_republished_without_generation(store, stage):
    store.fail_upload = "/video.mp4"
    with pytest.raises(OSError, match="upload interrupted"):
        cli.main(argv())
    failure = next(key for key in store.objects if key.endswith("failure.json"))
    recovery = failure.removesuffix("failure.json") + "output/"
    with pytest.raises(ValueError, match="checksum"):
        cli.main(argv())
    cli.main(["publish", "--input-path", recovery, "--output-path", argv()[-1]])
    cli.main(argv())
    assert len(stage[1]) == 1
    assert store.objects[argv()[-1] + "video.mp4"] == b"original GPU output"


def offline(*args):
    raise OSError("storage offline")


def test_failed_failure_publication_retains_local_output_and_original_error(
    monkeypatch, stage, tmp_path
):
    def broken(args, root, output):
        (output / "generation.log").write_text("original failure evidence")
        raise ValueError("original model failure")

    monkeypatch.setattr(cli, "_run", broken)
    monkeypatch.setattr("npa.workflows.physis_lang_recovery.publish", offline)
    with pytest.raises(ValueError, match="original model failure"):
        cli.main(argv(str(tmp_path / "unused")))
    output = stage[0][0] / "failure/output"
    assert (output / "generation.log").read_text() == "original failure evidence"
    assert not (output / "stage.json").exists()


def test_double_storage_failure_preserves_republishable_complete_output(
    monkeypatch, store, stage
):
    monkeypatch.setattr(cli, "publish", offline)
    monkeypatch.setattr("npa.workflows.physis_lang_recovery.publish", cli.publish)
    with pytest.raises(OSError, match="storage offline"):
        cli.main(argv())
    output = stage[0][0] / "failure/output"
    assert (output / "video.mp4").read_bytes() == b"original GPU output"
    assert (output / "stage.json").exists()
    artifacts.materialize(str(output), output)
    monkeypatch.setattr(cli, "publish", artifacts.publish)
    cli.main(["publish", "--input-path", str(output), "--output-path", argv()[-1]])
    cli.main(argv())
    assert len(stage[1]) == 1


def test_completed_destination_rejects_changed_request_before_generation(store, stage):
    cli.main(argv())
    changed = argv()
    changed[2] = "1"
    with pytest.raises(ValueError, match="different stage request"):
        cli.main(changed)
    assert len(stage[1]) == 1


@pytest.mark.parametrize("destination", ["s3", "local"])
def test_publication_is_idempotent_and_rejects_conflicting_bytes(
    store, tmp_path, destination
):
    root = tmp_path / "source"
    artifacts.write_json(root / "recipe.json", {"id": "same-run"})
    target = (
        "s3://example-bucket/artifacts/"
        if destination == "s3"
        else str(tmp_path / "published")
    )
    artifacts.publish(root, target)
    artifacts.publish(root, target)
    artifacts.write_json(root / "recipe.json", {"id": "another-run"})
    with pytest.raises(ValueError, match="conflict"):
        artifacts.publish(root, target)
    saved = artifacts.materialize(target, tmp_path / "readback")
    assert json.loads((saved / "recipe.json").read_text()) == {"id": "same-run"}


def test_competing_publisher_cannot_replace_a_reserved_manifest(
    store, monkeypatch, tmp_path
):
    root = tmp_path / "source"
    artifacts.write_json(root / "recipe.json", {"id": "our-run"})
    original = store.put_bytes_conditional

    def race(payload, uri, **kwargs):
        store.objects[uri] = b"a different publisher's manifest"
        return original(payload, uri, **kwargs)

    monkeypatch.setattr(store, "put_bytes_conditional", race)
    with pytest.raises(ValueError, match="conflicts"):
        artifacts.publish(root, "s3://example-bucket/artifacts/")
    assert len(store.objects) == 1


def test_failed_execution_publishes_partial_evidence_without_success(
    monkeypatch, store, stage
):
    def broken(args, root, output):
        (output / "generation.log").write_text("inference failed")
        raise RuntimeError("CUDA failure")

    monkeypatch.setattr(cli, "_run", broken)
    with pytest.raises(RuntimeError, match="CUDA failure"):
        cli.main(argv())
    assert store.objects
    assert all("-failed/" in key for key in store.objects)
    assert not any(key.endswith("stage.json") for key in store.objects)
    assert not any(root.exists() for root in stage[0])


def test_existing_symlink_cannot_redirect_local_recovery(tmp_path):
    source, destination, outside = (
        tmp_path / name for name in ("source", "destination", "outside")
    )
    artifacts.write_json(source / "clip/generation.json", {"status": "completed"})
    destination.mkdir()
    outside.mkdir()
    (destination / "clip").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        artifacts.publish(source, str(destination))
    assert not list(outside.iterdir())
