"""Bind actual encoded Arena-style frames to simulator capture and action evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.video_evidence import verify_capture_evidence
from npa.workbench.isaac_arena.simulator_video import (
    _MINIMUM_SETTLING_RENDERS,
    _RENDER_SETTINGS,
)


def _frame(column: int) -> np.ndarray:
    frame = np.full((240, 320, 3), [60, 80, 100], dtype=np.uint8)
    frame[90:114, column : column + 40] = [240, 180, 90]
    return frame


def _encode(path: Path, frames: list[np.ndarray], *, rate: int = 15) -> None:
    arguments = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    arguments += ["-s", "320x240", "-r", str(rate), "-i", "pipe:0", "-an"]
    arguments += [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(path),
    ]
    completed = subprocess.run(
        arguments,
        input=np.stack(frames).tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def _png_record(path: Path, frame: np.ndarray, step: int) -> dict:
    Image.fromarray(frame).save(path, format="PNG")
    return {
        "path": path.name,
        "action_step": step,
        "source_frame_index": step - 1 if step else None,
        "width": 320,
        "height": 240,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoded_rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
    }


def _write_sidecar(root: Path, payload: dict) -> None:
    (root / "simulator-video-evidence.json").write_text(json.dumps(payload))


@pytest.fixture
def capture(tmp_path: Path) -> tuple[Path, dict, dict, list[np.ndarray]]:
    frames = [_frame(20 + 3 * step) for step in range(20)]
    video = tmp_path / "raw.mp4"
    _encode(video, frames)
    payload = {
        "schema": "npa.isaac-arena.video-capture.v2",
        "capture_phase": "recorder_post_step_before_autoreset",
        "physics_clock": "native_physx_step_events_since_capture_setup",
        "captured_action_steps": 20,
        "initial": _png_record(tmp_path / "initial.png", _frame(17), 0),
        "terminals": [_png_record(tmp_path / "terminal.png", frames[-1], 20)],
        "rendering": {
            "mode": "RaytracedLighting",
            "legacy_mode_enabled": True,
            "rt2_enabled": False,
            "path_tracing_enabled": False,
            "antialiasing": "TAA",
            "dlss_execution_mode": "quality",
            "dl_denoiser_enabled": True,
            "frame_generation_enabled": False,
            "minimum_settling_renders": _MINIMUM_SETTLING_RENDERS,
            "stochastic_accumulation": False,
            "readback_phase": "after_final_accepted_render",
            "settings": dict(_RENDER_SETTINGS),
        },
        "physics_freeze_checks": [
            {
                "action_step": step,
                "physics_time_before": step / 50,
                "physics_time_after": step / 50,
                "physics_step_before": step,
                "physics_step_after": step,
                "native_physics_step_before": step,
                "native_physics_step_after": step,
                "state_sha256_before": hashlib.sha256(str(step).encode()).hexdigest(),
                "state_sha256_after": hashlib.sha256(str(step).encode()).hexdigest(),
                "render_calls": _MINIMUM_SETTLING_RENDERS + 1,
                "pre_settling_render_calls": 1,
                "settling_render_calls": _MINIMUM_SETTLING_RENDERS,
                "consecutive_ready_render_calls": _MINIMUM_SETTLING_RENDERS + 1,
                "stage_streaming_idle": True,
                "stage_assets_loaded": True,
                "nonblack_rgb": True,
            }
            for step in range(21)
        ],
    }
    _write_sidecar(tmp_path, payload)
    motion = {
        "progress_interval": {
            "start_action_step": 0,
            "end_action_step": 20,
            "total_action_steps": 20,
        },
        "video_capture": {
            "initial_action_step": 0,
            "action_steps": list(range(1, 21)),
            "terminal_action_step": 20,
        },
    }
    return video, payload, motion, frames


def test_capture_verifies_real_png_and_encoded_terminal_frame(capture) -> None:
    video, payload, motion, _ = capture
    original = video.read_bytes()
    result = verify_capture_evidence(
        video.parent, video, task_motion=motion, expected_steps=20
    )
    assert result["source_mp4_sha256"] == hashlib.sha256(original).hexdigest()
    assert video.read_bytes() == original
    assert result["initial"]["sha256"] == payload["initial"]["sha256"]
    assert (
        result["terminal"]["decoded_rgb_sha256"]
        == payload["terminals"][0]["decoded_rgb_sha256"]
    )
    comparison = result["terminal_frame_comparison"]
    assert comparison["source_frame_index"] == 19
    assert comparison["mean_absolute_luma_difference"] < 1


def test_capture_spans_full_episode_when_progress_threshold_crosses_early(
    capture,
) -> None:
    video, payload, motion, _ = capture
    motion["progress_interval"]["end_action_step"] = 10
    result = verify_capture_evidence(
        video.parent, video, task_motion=motion, expected_steps=20
    )
    assert result["captured_action_steps"] == 20
    assert result["terminal"]["action_step"] == 20
    assert result["terminal"]["sha256"] == payload["terminals"][0]["sha256"]


def test_short_raw_native_episode_keeps_every_action_frame(capture) -> None:
    video, _, motion, frames = capture
    _encode(video, frames, rate=50)
    result = verify_capture_evidence(
        video.parent, video, task_motion=motion, expected_steps=20
    )
    assert result["captured_action_steps"] == 20
    assert result["terminal_frame_comparison"]["source_frame_index"] == 19


def test_autoreset_frame_cannot_replace_successful_terminal_frame(capture) -> None:
    video, _, motion, frames = capture
    frames[-1] = _frame(17)
    _encode(video, frames)
    with pytest.raises(IsaacArenaError, match="terminal frame disagrees"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_consistent_hashes_cannot_hide_swapped_terminal_png(capture) -> None:
    video, payload, motion, _ = capture
    payload["terminals"][0] = _png_record(video.parent / "terminal.png", _frame(17), 20)
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="terminal frame disagrees"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_tampered_png_bytes_fail_even_before_frame_comparison(capture) -> None:
    video, _, motion, _ = capture
    Image.fromarray(_frame(17)).save(video.parent / "terminal.png", format="PNG")
    with pytest.raises(IsaacArenaError, match="PNG hash or dimensions disagree"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


@pytest.mark.parametrize("count", [19, 21])
def test_capture_counts_must_match_raw_video(capture, count: int) -> None:
    video, payload, motion, _ = capture
    payload["captured_action_steps"] = count
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="frame count disagrees"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_native_episode_action_count_is_independently_required(capture) -> None:
    video, _, motion, _ = capture
    with pytest.raises(IsaacArenaError, match="frame count disagrees"):
        verify_capture_evidence(
            video.parent, video, task_motion=motion, expected_steps=21
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_action_step", 1),
        ("terminal_action_step", 19),
        ("action_steps", list(range(1, 20))),
        ("action_steps", [1] * 20),
    ],
)
def test_hdf_action_steps_cannot_be_missing_repeated_or_shifted(
    capture, field, value
) -> None:
    video, _, motion, _ = capture
    motion["video_capture"][field] = value
    with pytest.raises(IsaacArenaError, match="action-step binding disagrees"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_terminal_png_requires_exact_task_success_action_step(capture) -> None:
    video, payload, motion, _ = capture
    payload["terminals"][0]["action_step"] = 19
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="frame action step disagrees"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_missing_png_cannot_be_replaced_by_sidecar_claim(capture) -> None:
    video, _, motion, _ = capture
    (video.parent / "initial.png").unlink()
    with pytest.raises(IsaacArenaError, match="missing simulator capture PNG"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_capture_after_reset_is_an_invalid_evidence_phase(capture) -> None:
    video, payload, motion, _ = capture
    payload["capture_phase"] = "after_autoreset"
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="schema or phase"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


@pytest.mark.parametrize(
    "field,value",
    [
        ("physics_time_after", 99.0),
        ("physics_step_after", 99),
        ("native_physics_step_after", 99),
        ("state_sha256_after", "f" * 64),
        ("stage_streaming_idle", False),
        ("stage_assets_loaded", False),
        ("nonblack_rgb", False),
        ("render_calls", _MINIMUM_SETTLING_RENDERS - 1),
        ("pre_settling_render_calls", 0),
        ("settling_render_calls", 4),
        ("consecutive_ready_render_calls", _MINIMUM_SETTLING_RENDERS),
        ("action_step", True),
    ],
)
def test_render_side_effects_and_unready_frames_cannot_qualify(capture, field, value):
    video, payload, motion, _ = capture
    payload["physics_freeze_checks"][10][field] = value
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="physics freeze evidence"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


@pytest.mark.parametrize("mutation", ["absent", "skipped", "repeated", "extra"])
def test_every_capture_including_initial_requires_its_own_freeze_check(
    capture, mutation
):
    video, payload, motion, _ = capture
    rows = payload["physics_freeze_checks"]
    if mutation == "absent":
        del payload["physics_freeze_checks"]
    elif mutation == "skipped":
        rows.pop(0)
    elif mutation == "repeated":
        rows[10] = rows[9]
    else:
        rows.append(rows[-1])
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="physics freeze evidence"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_allblack_initial_png_cannot_qualify_with_consistent_hashes(capture):
    video, payload, motion, frames = capture
    payload["initial"] = _png_record(
        video.parent / "initial.png", np.zeros_like(frames[0]), 0
    )
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="entirely black"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_renderer_readback_must_confirm_temporal_antialiasing(capture):
    video, payload, motion, _ = capture
    payload["rendering"]["settings"]["/rtx/post/aa/op"] = 2
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="renderer readback"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


def test_renderer_readback_must_reject_generated_frames(capture):
    video, payload, motion, _ = capture
    payload["rendering"]["settings"]["/rtx-transient/dlssg/enabled"] = True
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="renderer readback"):
        verify_capture_evidence(video.parent, video, task_motion=motion)


@pytest.mark.parametrize(
    "field", ["physics_time", "native_physics_step", "physics_step"]
)
def test_inactive_physics_observation_cannot_claim_rendering_is_frozen(capture, field):
    video, payload, motion, _ = capture
    for row in payload["physics_freeze_checks"]:
        row[f"{field}_before"] = 0.0 if field == "physics_time" else 0
        row[f"{field}_after"] = row[f"{field}_before"]
    _write_sidecar(video.parent, payload)
    with pytest.raises(IsaacArenaError, match="did not advance between real actions"):
        verify_capture_evidence(video.parent, video, task_motion=motion)
