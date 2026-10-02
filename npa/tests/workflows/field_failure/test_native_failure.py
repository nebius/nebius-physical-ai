"""Prove failed native artifacts survive outer adapter cleanup without claiming success."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType

import pytest

from npa.workflows.field_failure import native_artifacts, native_policy


@pytest.fixture
def native_execution(monkeypatch):
    uploaded = {}

    class Storage:
        def put_bytes_conditional(self, data, uri, *, if_none_match):
            assert if_none_match is True and uri not in uploaded
            uploaded[uri] = data

    monkeypatch.setattr(native_artifacts, "_storage", lambda: Storage())
    stages = ModuleType("npa.workflows.navigation.stages")
    stages.prepare = lambda *args: None
    monkeypatch.setitem(sys.modules, stages.__name__, stages)
    image = "registry.example.invalid/navigation@sha256:" + "a" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    request = {
        "bundle_sha256": "b" * 64,
        "adapter": {
            "entrypoint": "fixture:train",
            "source_sha256": "c" * 64,
            "runtime_image": image,
        },
        "attempt_id": "fixture-attempt",
        "inputs_sha256": "d" * 64,
        "protocol": {"sha256": "e" * 64},
        "output_prefix": "s3://fixture/runs/fixture/attempt-fixture/",
    }
    return stages, request, {"navigation_image": image}, uploaded


def _failed_native(error, write_output=True):
    def run(stage, source, output):
        if write_output:
            destination = Path(output)
            destination.mkdir()
            (destination / "runtime.log").write_text(
                "synthetic native failure detail\n"
            )
            (destination / "failure.json").write_text(
                json.dumps({"status": "failed", "stage": stage})
            )
            (destination / "checksums.json").write_text("{}")
        raise error

    return run


@pytest.mark.parametrize(
    "stage,name",
    [
        ("train", "trained-0"),
        ("evaluate-checkpoint", "baseline-0-replay-output"),
        ("evaluate-checkpoint", "candidate-0-replay-output"),
        ("evaluate-checkpoint", "evaluation-0"),
    ],
)
def test_native_failure_is_retained_before_adapter_cleanup(
    native_execution, tmp_path, stage, name
):
    stages, request, protocol, uploaded = native_execution
    original = FileNotFoundError("synthetic missing native completion")
    stages.run_stage = _failed_native(original)
    with tempfile.TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        with pytest.raises(FileNotFoundError) as caught:
            native_policy._execute(
                request, stage, root / "source", root / name, protocol
            )
        assert caught.value is original
    assert not root.exists()
    prefix = request["output_prefix"] + name + "-failure"
    record = json.loads(uploaded[prefix + ".json"])
    archive = uploaded[prefix + ".tar"]
    assert record["schema_version"] == "npa.field-failure.native-failure.v1"
    assert record["status"] == "failed" and record["native_runtime_verified"] is False
    assert record["native_stage"] == stage
    assert record["inputs_sha256"] == request["inputs_sha256"]
    assert record["protocol_sha256"] == request["protocol"]["sha256"]
    assert record["artifacts"] == {
        "uri": prefix + ".tar",
        "sha256": hashlib.sha256(archive).hexdigest(),
    }
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        assert (
            stream.extractfile("runtime.log").read()
            == b"synthetic native failure detail\n"
        )
        assert json.load(stream.extractfile("failure.json"))["stage"] == stage
        assert "evaluation.json" not in stream.getnames()
    assert set(uploaded) == {prefix + ".tar", prefix + ".json"}


def test_native_failure_without_output_does_not_fabricate_artifacts(
    native_execution, tmp_path
):
    stages, request, protocol, uploaded = native_execution
    original = RuntimeError("synthetic failure before native output")
    stages.run_stage = _failed_native(original, write_output=False)
    with pytest.raises(RuntimeError) as caught:
        native_policy._execute(
            request, "train", tmp_path / "source", tmp_path / "trained-0", protocol
        )
    assert caught.value is original and not uploaded


def test_successful_native_execution_does_not_publish_failure(
    native_execution, tmp_path
):
    stages, request, protocol, uploaded = native_execution
    report = {"synthetic": "successful native report"}
    stages.run_stage = lambda *args: report
    result = native_policy._execute(
        request, "train", tmp_path / "source", tmp_path / "trained-0", protocol
    )
    assert result is report and not uploaded


def test_failed_publication_keeps_original_native_exception_chain(
    native_execution, tmp_path, monkeypatch
):
    stages, request, protocol, uploaded = native_execution
    original = FileNotFoundError("synthetic missing native completion")
    stages.run_stage = _failed_native(original)

    class BrokenStorage:
        def put_bytes_conditional(self, *args, **kwargs):
            raise OSError("synthetic conditional storage failure")

    monkeypatch.setattr(native_artifacts, "_storage", lambda: BrokenStorage())
    with pytest.raises(OSError, match="conditional storage failure") as caught:
        native_policy._execute(
            request, "train", tmp_path / "source", tmp_path / "trained-0", protocol
        )
    assert caught.value.__context__ is original and not uploaded
