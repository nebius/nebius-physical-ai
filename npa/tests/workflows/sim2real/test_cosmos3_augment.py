"""Focused contracts for canonical Sim2Real Cosmos3 structural augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows.sim2real.cosmos3_augment import (
    _assert_guarded_transfer,
    run_cosmos3_structural_augmentation,
    select_source_frames,
)
from npa.workflows.sim2real.models import Sim2RealLoopError
from npa.workflows.sim2real_envgen import frame_uris_from_transfer_manifest


class _Store:
    def __init__(self) -> None:
        self.downloaded: list[str] = []
        self.uploaded: list[str] = []

    def download_directory(self, uri: str, destination: str) -> None:
        self.downloaded.append(uri)
        target = Path(destination)
        target.mkdir(parents=True)
        for index in range(6):
            (target / f"camera-{index}.png").write_bytes(f"frame-{index}".encode())

    def upload_file(self, local: str, uri: str) -> None:
        assert Path(local).is_file()
        self.uploaded.append(uri)


def test_select_source_frames_prefers_complete_primary_camera_sequence(
    tmp_path: Path,
) -> None:
    for index in range(6):
        (tmp_path / f"camera-{index}.png").write_bytes(b"camera")
        (tmp_path / f"frame-{index}.png").write_bytes(b"fallback")

    selected = select_source_frames(tmp_path)

    assert [path.name for path in selected] == [
        f"camera-{index}.png" for index in range(6)
    ]


def test_select_source_frames_rejects_ambiguous_or_short_sequences(
    tmp_path: Path,
) -> None:
    for index in range(5):
        (tmp_path / f"camera-{index}.png").write_bytes(b"camera")
    with pytest.raises(Sim2RealLoopError, match="at least six"):
        select_source_frames(tmp_path)

    (tmp_path / "camera-5.png").write_bytes(b"camera")
    duplicate = tmp_path / "nested"
    duplicate.mkdir()
    (duplicate / "camera-5.png").write_bytes(b"camera")
    with pytest.raises(Sim2RealLoopError, match="uniquely indexed"):
        select_source_frames(tmp_path)


def test_canonical_stage3_publishes_complete_guarded_frames_consumed_by_envgen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Require actual Cosmos3 API wiring and an indexed full-frame handoff."""

    from npa.workbench.cosmos import generate
    from npa.workflows import paidf_cosmos3_media
    from npa.workflows.sim2real import cosmos3_augment
    from npa.workflows.sim2real import workflow_io

    store = _Store()
    written: dict[str, dict] = {}

    def encode(_frames: list[Path], destination: Path) -> None:
        destination.write_bytes(b"encoded-source")

    def prepare(source: Path, destination: Path, fps: int) -> dict:
        assert source.is_file() and fps == 24
        destination.write_bytes(b"prepared-source")
        return {"decoded_frames": 6, "fps": fps}

    def generate_transfer(**kwargs):
        assert kwargs["mode"] == "video2video"
        assert kwargs["checkpoint"] == "Cosmos3-Nano"
        assert kwargs["no_guardrails"] is False
        assert kwargs["transfer"].fps == 24
        assert kwargs["transfer"].chunk_frames == 93
        assert kwargs["transfer"].control_guidance == 1.5
        output = tmp_path / "generated.mp4"
        control = tmp_path / "edge-control.mkv"
        output.write_bytes(b"generated-video")
        control.write_bytes(b"edge-control")
        return {
            "output_path": str(output),
            "guardrail_state": {"status": "passed", "effective": True},
            "structural_transfer": {
                "text_guardrail_passed": True,
                "video_guardrail_passed": True,
                "guardrail_postprocessing_applied": True,
                "control_path": str(control),
            },
            "native_model_selection": {"checkpoint": "Cosmos3-Nano"},
            "input_spec": {"mode": "video2video"},
        }

    def decode(_video: Path, destination: Path, expected: int) -> list[Path]:
        assert expected == 6
        destination.mkdir()
        frames: list[Path] = []
        for index in range(expected):
            frame = destination / f"frame-{index:05d}.png"
            frame.write_bytes(f"generated-{index}".encode())
            frames.append(frame)
        return frames

    monkeypatch.setattr(cosmos3_augment, "encode_source_video", encode)
    monkeypatch.setattr(cosmos3_augment, "decode_generated_frames", decode)
    monkeypatch.setattr(generate, "run_cosmos3_generate", generate_transfer)
    monkeypatch.setattr(paidf_cosmos3_media, "prepare_reference", prepare)
    monkeypatch.setattr(
        paidf_cosmos3_media, "probe_video", lambda _path: {"decoded_frames": 6}
    )
    monkeypatch.setattr(
        paidf_cosmos3_media,
        "verify_pair",
        lambda _source, _generated, fps: {"decoded_frames": 6, "fps": fps},
    )
    monkeypatch.setattr(
        workflow_io,
        "write_json",
        lambda uri, payload, **_kwargs: written.setdefault(uri, payload),
    )

    result = run_cosmos3_structural_augmentation(
        input_uri="s3://seed/primary",
        root_uri="s3://run/root",
        run_id="fresh-cosmos3-run",
        client=store,
    )

    manifest = result["manifest"]
    assert store.downloaded == ["s3://seed/primary/"]
    assert manifest["schema"] == "npa.cosmos3.structural-transfer.v1"
    assert manifest["mode"] == "cosmos3_nano_video2video_edge_gpu"
    assert manifest["full_source_coverage"] == {
        "selected_source_frames": 6,
        "prepared_frames": 6,
        "generated_frames": 6,
        "decoded_envgen_frames": 6,
        "complete": True,
    }
    assert [
        record["ordinal"] for record in manifest["source_sequence"]["frames"]
    ] == list(range(6))
    assert [record["name"] for record in manifest["source_sequence"]["frames"]] == [
        f"camera-{index}.png" for index in range(6)
    ]
    assert len(manifest["frames"]) == 6
    assert manifest["downstream_consumers"] == [
        "stage_04_envgen_raw",
        "stage_05_split_scenario_lineage",
        "stage_08_rollout_context",
    ]
    assert result["result_uri"] in written
    assert len(store.uploaded) == 10  # source/prepared/output/control + all frames

    manifest_path = tmp_path / "cosmos3-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert frame_uris_from_transfer_manifest(str(manifest_path)) == tuple(
        f"s3://run/root/augment/frames/frame-{index:05d}.png" for index in range(6)
    )


def test_guarded_transfer_rejects_missing_effective_video_guardrail(
    tmp_path: Path,
) -> None:
    control = tmp_path / "edges.mkv"
    control.write_bytes(b"edge")
    result = {
        "guardrail_state": {"status": "passed", "effective": True},
        "structural_transfer": {
            "text_guardrail_passed": True,
            "video_guardrail_passed": False,
            "guardrail_postprocessing_applied": True,
            "control_path": str(control),
        },
    }

    with pytest.raises(Sim2RealLoopError, match="guarded structural-transfer"):
        _assert_guarded_transfer(result)
