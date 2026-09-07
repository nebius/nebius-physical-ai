from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows.sim2real.cosmos_transfer_stage import (
    _task_conditioned_transfer_input,
    run_cosmos_transfer_component,
)
from npa.workflows.sim2real.models import Sim2RealLoopError


class _RecordingStorage:
    def __init__(self) -> None:
        self.uploads: list[str] = []

    def upload_file(self, local: str, uri: str) -> str:
        assert Path(local).is_file()
        self.uploads.append(uri)
        return uri


class _SeedStorage:
    def download_directory(self, _uri: str, local: str) -> None:
        frames = Path(local) / "frames" / "trajectory-000"
        frames.mkdir(parents=True)
        for index in range(4):
            (frames / f"frame-{index:04d}.png").write_bytes(b"png")
        (frames / "frame-preview.png").write_bytes(b"not-numbered")


def test_task_conditioned_transfer_accepts_numbered_seed_contract_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(argv: list[str], **_kwargs: object) -> None:
        Path(argv[-1]).write_bytes(b"video")

    monkeypatch.setattr(
        "npa.workflows.sim2real.cosmos_transfer_stage.subprocess.run", fake_run
    )
    calls: list[dict[str, object]] = []

    def run_transfer(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "executed"}

    result, fixture = _task_conditioned_transfer_input(
        _SeedStorage(), "s3://unit/seed/", "run-1", run_transfer
    )

    assert result == {"status": "executed"}
    assert fixture is not None
    assert fixture["source_frame_count"] == 4
    assert calls[0]["prompt"].startswith("A photorealistic Franka Panda robot")


def _real_result(*, frames: object = None, frame_count: int = 1) -> dict:
    actual_frames = (
        [
            {
                "frame_id": "frame-00000",
                "uri": "s3://unit/run/augment/frames/frame-00000.png",
            }
        ]
        if frames is None
        else frames
    )
    return {
        "augmented_video_uri": "s3://unit/run/augment/video/augmented.mp4",
        "frame_count": frame_count,
        "frames": actual_frames,
        "video_bytes": 42,
        "spec": "task-conditioned.json",
        "input_conditioned": True,
    }


@pytest.mark.parametrize(
    ("output_uri", "expected_result"),
    [
        (
            "s3://unit/run/augment/cosmos2-transfer-result.json",
            "s3://unit/run/augment/cosmos2-transfer-result.json",
        ),
        (
            "s3://unit/run/augment/",
            "s3://unit/run/augment/cosmos2-transfer-result.json",
        ),
    ],
)
def test_transfer_result_object_and_directory_publish_manifest_at_same_prefix(
    monkeypatch: pytest.MonkeyPatch,
    output_uri: str,
    expected_result: str,
) -> None:
    storage = _RecordingStorage()
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment", lambda: storage
    )

    result = run_cosmos_transfer_component(
        input_uri="s3://unit/seed/",
        output_uri=output_uri,
        augmented_frames_uri="s3://unit/run/augment/frames/",
        real_runner=lambda *_args: _real_result(),
    )

    assert result["manifest"]["mode"] == "cosmos_transfer2.5_gpu"
    assert "s3://unit/run/augment/manifest.json" in storage.uploads
    assert expected_result in storage.uploads


@pytest.mark.parametrize(
    ("frames", "frame_count", "message"),
    [
        (
            [{"frame_id": "frame-00000", "uri": "s3://unit/elsewhere/frame.png"}],
            1,
            "outside its declared output prefix",
        ),
        (
            [{"frame_id": "", "uri": "s3://unit/run/augment/frames/f.png"}],
            1,
            "malformed frame",
        ),
        ("not-a-list", 1, "non-empty exact frames list"),
        (
            [
                {
                    "frame_id": "frame-00000",
                    "uri": "s3://unit/run/augment/frames/frame-00000.png",
                }
            ],
            2,
            "frame_count does not match",
        ),
        (
            [
                {
                    "frame_id": "frame-00000",
                    "uri": "s3://unit/run/augment/frames/frame-00000.png",
                },
                {
                    "frame_id": "frame-00000",
                    "uri": "s3://unit/run/augment/frames/frame-00000.png",
                },
            ],
            2,
            "duplicate frame lineage",
        ),
    ],
)
def test_real_transfer_frame_lineage_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    frames: object,
    frame_count: int,
    message: str,
) -> None:
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda: _RecordingStorage(),
    )
    with pytest.raises(Sim2RealLoopError, match=message):
        run_cosmos_transfer_component(
            input_uri="s3://unit/seed/",
            output_uri="s3://unit/run/augment/",
            augmented_frames_uri="s3://unit/run/augment/frames/",
            real_runner=lambda *_args: _real_result(
                frames=frames, frame_count=frame_count
            ),
        )


def test_required_real_runner_returning_none_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "1")
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda: _RecordingStorage(),
    )
    with pytest.raises(Sim2RealLoopError, match="attempted to fall back"):
        run_cosmos_transfer_component(
            input_uri="s3://unit/seed/",
            output_uri="s3://unit/run/augment/",
            augmented_frames_uri="s3://unit/run/augment/frames/",
            real_runner=lambda *_args: None,
        )
