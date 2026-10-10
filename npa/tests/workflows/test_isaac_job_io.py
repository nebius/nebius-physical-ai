from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError, EndpointConnectionError

from npa.workflows.sim2real import isaac_job_io


class _FlakyS3:
    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = failures
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, source: str, bucket: str, key: str) -> None:
        self.calls.append((source, bucket, key))
        if self.failures:
            raise self.failures.pop(0)


def _client_error(status: int, code: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "structured test error"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "PutObject",
    )


def test_upload_recovers_from_typed_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    s3 = _FlakyS3([EndpointConnectionError(endpoint_url="https://storage.invalid")])
    sleeps: list[float] = []
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(isaac_job_io.time, "sleep", sleeps.append)

    isaac_job_io.upload(source, "s3://bucket/prefix/artifact.bin")

    assert len(s3.calls) == 2
    assert sleeps == [2.0]
    output = capsys.readouterr().out
    assert "classification=EndpointConnectionError" in output
    assert "state=retrying" in output


@pytest.mark.parametrize(
    ("status", "code"),
    [(503, "ServiceUnavailable"), (429, "SlowDown"), (400, "RequestTimeout")],
)
def test_upload_recovers_from_structured_service_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    code: str,
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    s3 = _FlakyS3([_client_error(status, code)])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(isaac_job_io.time, "sleep", lambda _delay: None)

    isaac_job_io.upload(source, "s3://bucket/prefix/artifact.bin")

    assert len(s3.calls) == 2


def test_upload_fails_closed_for_nonretryable_client_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    error = _client_error(403, "AccessDenied")
    s3 = _FlakyS3([error])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(
        isaac_job_io.time,
        "sleep",
        lambda _delay: pytest.fail("nonretryable errors must not sleep"),
    )

    with pytest.raises(ClientError) as raised:
        isaac_job_io.upload(source, "s3://bucket/prefix/artifact.bin")

    assert raised.value is error
    assert len(s3.calls) == 1


@pytest.mark.parametrize("status", [403, 503])
def test_managed_transfer_wrapper_retains_structured_retry_decision(
    monkeypatch, tmp_path, status
):
    source = tmp_path / "camera.png"
    source.write_bytes(b"camera")
    error = S3UploadFailedError("managed transfer failed")
    error.__context__ = _client_error(
        status, "AccessDenied" if status == 403 else "SlowDown"
    )
    s3 = _FlakyS3([error])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(isaac_job_io.time, "sleep", lambda delay: None)
    if status == 503:
        isaac_job_io.upload(source, "s3://bucket/camera.png")
        assert len(s3.calls) == 2
    else:
        with pytest.raises(S3UploadFailedError) as raised:
            isaac_job_io.upload(source, "s3://bucket/camera.png")
        assert raised.value is error
        assert len(s3.calls) == 1


def test_unstructured_transfer_message_does_not_trigger_retry(monkeypatch, tmp_path):
    source = tmp_path / "camera.png"
    source.write_bytes(b"camera")
    error = S3UploadFailedError("ServiceUnavailable retry this upload")
    s3 = _FlakyS3([error])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    with pytest.raises(S3UploadFailedError):
        isaac_job_io.upload(source, "s3://bucket/camera.png")
    assert len(s3.calls) == 1


def test_upload_tree_retries_current_file_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "camera-0001.png").write_bytes(b"one")
    (tmp_path / "camera-0002.png").write_bytes(b"two")
    s3 = _FlakyS3([_client_error(503, "ServiceUnavailable")])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(isaac_job_io.time, "sleep", lambda _delay: None)

    isaac_job_io.upload_tree(tmp_path, "s3://bucket/run/renders")

    assert [call[2] for call in s3.calls] == [
        "run/renders/camera-0001.png",
        "run/renders/camera-0001.png",
        "run/renders/camera-0002.png",
    ]
    output = capsys.readouterr().out
    assert "operation=upload-tree state=progress files=1 bytes=3" in output
    assert "UPLOADED_TREE uri=s3://bucket/run/renders files=2" in output


def test_retry_delay_configuration_must_be_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_S3_IO_RETRY_BASE_SECONDS", "0")

    with pytest.raises(ValueError, match="must be positive"):
        isaac_job_io._retry_delay(1)


def test_upload_tree_uses_concurrent_workers_and_relative_keys(monkeypatch, tmp_path):
    rendezvous = Barrier(3)
    uploaded = []
    for name in ("one.png", "side/two.png", "overhead/three.png"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"camera")

    class Storage:
        def upload_file(self, source, bucket, key):
            rendezvous.wait(timeout=5)
            uploaded.append((bucket, key, Path(source).read_bytes()))

    monkeypatch.setattr(isaac_job_io, "_s3", Storage)
    isaac_job_io.upload_tree(tmp_path, "s3://bucket", max_workers=3)
    assert sorted(uploaded) == [
        ("bucket", "one.png", b"camera"),
        ("bucket", "overhead/three.png", b"camera"),
        ("bucket", "side/two.png", b"camera"),
    ]


def test_terminal_upload_failure_stops_sibling_transport_retries(monkeypatch, tmp_path):
    retry_started = Event()
    terminal = _client_error(403, "AccessDenied")
    attempts = []
    (tmp_path / "retry.png").write_bytes(b"camera")
    (tmp_path / "rejected.png").write_bytes(b"camera")

    class Storage:
        def upload_file(self, source, bucket, key):
            attempts.append(key)
            if key.endswith("retry.png"):
                retry_started.set()
                raise EndpointConnectionError(endpoint_url="https://storage.invalid")
            assert retry_started.wait(timeout=5)
            raise terminal

    monkeypatch.setattr(isaac_job_io, "_s3", Storage)
    monkeypatch.setenv("NPA_S3_IO_RETRY_BASE_SECONDS", "30")
    with pytest.raises(ClientError) as raised:
        isaac_job_io.upload_tree(tmp_path, "s3://bucket/renders", max_workers=2)
    assert raised.value is terminal
    assert sorted(attempts) == ["renders/rejected.png", "renders/retry.png"]


def test_capture_publishes_completion_metadata_after_all_camera_files(
    monkeypatch, tmp_path
):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "camera.png").write_bytes(b"camera")
    (frames / "cloud.npz").write_bytes(b"pointcloud")
    metadata = tmp_path / "metrics.json"
    metadata.write_text("{}")
    s3 = _FlakyS3([])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)

    isaac_job_io.upload_capture(
        frames, "s3://bucket/renders", metadata, "s3://bucket/metrics.json"
    )

    assert sorted(call[2] for call in s3.calls[:-1]) == [
        "renders/camera.png",
        "renders/cloud.npz",
    ]
    assert s3.calls[-1][2] == "metrics.json"


@pytest.mark.parametrize("missing_tree", [False, True])
def test_capture_never_publishes_metadata_for_incomplete_camera_tree(
    monkeypatch, tmp_path, missing_tree
):
    frames = tmp_path / "frames"
    frames.mkdir()
    if not missing_tree:
        (frames / "camera.png").write_bytes(b"camera")
    metadata = tmp_path / "metrics.json"
    metadata.write_text("{}")
    s3 = _FlakyS3([_client_error(403, "AccessDenied")])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)

    with pytest.raises((ClientError, RuntimeError)):
        isaac_job_io.upload_capture(
            frames, "s3://bucket/renders", metadata, "s3://bucket/metrics.json"
        )
    assert all(call[2] != "metrics.json" for call in s3.calls)


def test_capture_without_cameras_uploads_only_metadata(monkeypatch, tmp_path):
    metadata = tmp_path / "metrics.json"
    metadata.write_text("{}")
    s3 = _FlakyS3([])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    isaac_job_io.upload_capture(
        tmp_path / "absent", "", metadata, "s3://bucket/metrics.json"
    )
    assert [call[2] for call in s3.calls] == ["metrics.json"]


def _native_upload_namespace(exit_process):
    return {
        "os": SimpleNamespace(
            _exit=exit_process,
            environ={
                "EVAL_RENDERS_S3": "s3://bucket/renders",
                "EVAL_OUT_S3": "s3://bucket/metrics.json",
            },
        ),
        "sys": SimpleNamespace(stdout=io.StringIO(), stderr=io.StringIO()),
        "json": json,
        "open": lambda *args: io.StringIO(),
        "checkpoint_provenance": lambda: {},
        "trained": True,
        "CAMERA_VIEWS": [],
        "SIM_DEVICE": "cuda:0",
        "CAPTURE_WIDTH": 640,
        "CAPTURE_HEIGHT": 480,
        "CAPTURE_STRIDE": 1,
        "STEPS": 32,
        "HORIZON_STEPS": 300,
        "CAPTURE_STEPS": [0],
        "SAMPLE_STEPS": [0],
        "PNG_COMPRESS_LEVEL": 1,
        "CAPTURE_FPS": 30,
        "FRAMES_DIR": "/tmp/rollwork/frames",
        "OUT_S3": "s3://bucket/rollouts",
        "OUT": "/tmp/evalwork/metrics.json",
    }


def _native_upload_invocation(kind, namespace):
    from npa.workflows.sim2real import byo_isaac_eval, byo_isaac_policy_rollout

    if kind == "rollout":
        nodes = ast.parse(byo_isaac_policy_rollout.ISAAC_ROLLOUT_SCRIPT).body
        function = next(
            node
            for node in nodes
            if isinstance(node, ast.FunctionDef) and node.name == "upload_and_exit"
        )
        exec(compile(ast.Module([function], []), "<native-rollout>", "exec"), namespace)

        def invoke():
            namespace["upload_and_exit"]([], "test")
    else:
        nodes = ast.parse(byo_isaac_eval.ISAAC_EVAL_SCRIPT).body
        index = next(
            i
            for i, node in enumerate(nodes)
            if isinstance(node, ast.Try)
            and any(
                isinstance(item, ast.ImportFrom)
                and item.module == "npa.workflows.sim2real.isaac_job_io"
                for item in node.body
            )
        )
        code = compile(ast.Module(nodes[index:], []), "<native-evaluation>", "exec")

        def invoke():
            exec(code, namespace)

    return invoke


@pytest.mark.parametrize("kind", ["rollout", "evaluation"])
@pytest.mark.parametrize("upload_fails", [False, True])
def test_native_camera_scripts_exit_unsuccessfully_when_upload_fails(
    monkeypatch, kind, upload_fails
):
    calls = []

    def publish(*args):
        calls.append(args)
        if upload_fails:
            raise _client_error(403, "AccessDenied")

    def exit_process(code):
        raise SystemExit(code)

    monkeypatch.setattr(isaac_job_io, "upload_capture", publish)
    namespace = _native_upload_namespace(exit_process)
    invoke = _native_upload_invocation(kind, namespace)
    with pytest.raises(SystemExit) as raised:
        invoke()
    assert raised.value.code == (1 if upload_fails else 0)
    assert len(calls) == 1
