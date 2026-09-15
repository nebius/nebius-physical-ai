"""Exercise Arena's visual gate on independently encoded adversarial MP4s."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.video_evidence import (
    EVIDENCE_FILTER,
    EVIDENCE_PLAYBACK_RATE,
    denoise_mp4,
    probe_mp4,
)

_TASK_REGION = {
    "name": "microwave_door_workspace",
    "left": 0.25,
    "top": 0.3,
    "right": 0.7,
    "bottom": 0.95,
}


def _encode(path: Path, frames: np.ndarray, *, rate: int = 15) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            "320x240",
            "-r",
            str(rate),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ],
        input=frames.astype(np.uint8).tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def _scene(count: int, *, moving: bool = False) -> np.ndarray:
    rows, columns = np.indices((240, 320))
    background = np.full((240, 320), 64, dtype=np.float32)
    background[rows > 185] = 110
    background[(columns > 205) & (rows < 170) & (rows > 35)] = 160
    frames = np.repeat(background[None], count, axis=0)
    texture = 170 + ((rows[:70, :65] // 9 + columns[:70, :65] // 7) % 2) * 55
    for index in range(count):
        column = 25 + (index * 2 if moving else 0)
        frames[index, 80:150, column : column + 65] = texture
    return frames


def _add_noise(
    frames: np.ndarray, kind: str, seed: int, grain_size: int = 10
) -> np.ndarray:
    random = np.random.default_rng(seed)
    if kind == "fine":
        noise = random.normal(0, 60, frames.shape)
    elif kind == "coarse":
        coarse = random.normal(
            0, 45, (len(frames), 240 // grain_size, 320 // grain_size)
        )
        noise = np.repeat(np.repeat(coarse, grain_size, axis=1), grain_size, axis=2)
    elif kind == "flicker":
        noise = random.uniform(-35, 35, (len(frames), 1, 1))
    else:
        raise AssertionError(kind)
    return np.clip(frames + noise, 0, 255)


@pytest.mark.parametrize("kind", ["fine", "coarse", "flicker"])
@pytest.mark.parametrize("seed", [7, 19, 43])
def test_static_scene_with_strong_render_noise_fails(
    tmp_path: Path, kind: str, seed: int
) -> None:
    frames = _add_noise(_scene(45), kind, seed)
    # Every adversary would satisfy the retired raw-pixel motion test.
    differences = np.abs(frames[1:] - frames[:-1])
    assert np.mean(differences) > 5
    assert np.mean(differences >= 6) > 0.3
    video = tmp_path / "noisy-static.mp4"
    _encode(video, frames)
    with pytest.raises(IsaacArenaError, match="noise-resistant coherent scene motion"):
        probe_mp4(video)


@pytest.mark.parametrize("kind", [None, "fine", "coarse", "flicker"])
def test_translating_object_passes_with_noise(tmp_path: Path, kind: str | None) -> None:
    frames = _scene(45, moving=True)
    if kind:
        frames = _add_noise(frames, kind, 7)
    video = tmp_path / "moving.mp4"
    _encode(video, frames)
    metadata = probe_mp4(
        video,
        evidence_interval={
            "start_action_step": 1,
            "end_action_step": 45,
            "total_action_steps": 45,
        },
    )
    assert metadata["frame_count"] == 45
    assert metadata["motion"]["meaningful"] is True
    assert metadata["motion"]["changed_frame_pairs"] >= 2
    assert metadata["motion"]["max_coherent_blocks"] >= 2
    samples = metadata["frame_evidence"]["samples"]
    assert samples["first"]["source_frame_index"] == 0
    assert samples["last"]["source_frame_index"] == 44
    assert samples["first"]["timestamp_seconds"] == 0
    assert samples["last"]["timestamp_seconds"] == pytest.approx(44 / 15, abs=0.000001)
    assert (
        samples["first"]["decoded_luma_sha256"]
        != samples["last"]["decoded_luma_sha256"]
    )


def test_motion_outside_task_interval_cannot_supply_evidence(tmp_path: Path) -> None:
    frames = _scene(75)
    frames[:25] = _scene(25, moving=True)
    frames[50:] = _scene(25, moving=True)
    video = tmp_path / "wrong-interval.mp4"
    _encode(video, frames)
    assert probe_mp4(video)["motion"]["meaningful"]
    with pytest.raises(IsaacArenaError, match="noise-resistant coherent scene motion"):
        probe_mp4(
            video,
            evidence_interval={
                "start_action_step": 29,
                "end_action_step": 46,
                "total_action_steps": 75,
            },
        )


def test_coherent_motion_before_progress_cannot_satisfy_overlap(
    tmp_path: Path,
) -> None:
    frames = _scene(45)
    frames[:20] = _scene(20, moving=True)
    for index in range(29, 45):
        frames[index, 100:140, 100:140] += (index - 28) * 3
    video = tmp_path / "motion-before-progress.mp4"
    _encode(video, frames)
    with pytest.raises(IsaacArenaError, match="does not overlap native task progress"):
        probe_mp4(
            video,
            evidence_interval={
                "start_action_step": 0,
                "end_action_step": 45,
                "total_action_steps": 45,
            },
            progress_interval={
                "start_action_step": 30,
                "end_action_step": 45,
                "total_action_steps": 45,
            },
            progress_signal="monotonic_structural_change",
            progress_region=_TASK_REGION,
            progress_association_radius_fraction=0.03,
        )


def test_disjoint_motion_and_progress_ramp_cannot_qualify(tmp_path: Path) -> None:
    frames = _scene(45, moving=True)
    for index in range(29, 45):
        frames[index, 170:210, 200:240] += (index - 28) * 3
    video = tmp_path / "disjoint-motion-and-progress.mp4"
    _encode(video, np.clip(frames, 0, 255))
    with pytest.raises(IsaacArenaError, match="spatially disjoint"):
        probe_mp4(
            video,
            evidence_interval={
                "start_action_step": 0,
                "end_action_step": 45,
                "total_action_steps": 45,
            },
            progress_interval={
                "start_action_step": 30,
                "end_action_step": 45,
                "total_action_steps": 45,
            },
            progress_signal="monotonic_structural_change",
            progress_region=_TASK_REGION,
            progress_association_radius_fraction=0.03,
        )


def test_short_task_interval_is_not_expanded_to_surrounding_motion(
    tmp_path: Path,
) -> None:
    video = tmp_path / "short-interval.mp4"
    _encode(video, _scene(45, moving=True))
    with pytest.raises(IsaacArenaError, match="interval is too short"):
        probe_mp4(
            video,
            evidence_interval={
                "start_action_step": 20,
                "end_action_step": 25,
                "total_action_steps": 45,
            },
        )


def test_action_count_mismatch_fails_instead_of_proportional_mapping(
    tmp_path: Path,
) -> None:
    video = tmp_path / "mismatch.mp4"
    _encode(video, _scene(45, moving=True))
    with pytest.raises(IsaacArenaError, match="frame count does not match"):
        probe_mp4(
            video,
            evidence_interval={
                "start_action_step": 20,
                "end_action_step": 45,
                "total_action_steps": 90,
            },
        )


def test_late_motion_is_decoded_after_thirty_seconds_with_exact_mapping(
    tmp_path: Path,
) -> None:
    frames = _scene(510)
    frames[465:] = _scene(45, moving=True)
    video = tmp_path / "late-motion.mp4"
    _encode(video, frames)
    metadata = probe_mp4(
        video,
        evidence_interval={
            "start_action_step": 466,
            "end_action_step": 510,
            "total_action_steps": 510,
        },
    )
    interval = metadata["motion"]["analysis_interval"]
    assert interval["source_frame_indices"] == {"start": 465, "end_exclusive": 510}
    assert metadata["motion"]["decoded_samples"] == 45
    samples = metadata["frame_evidence"]["samples"]
    assert samples["first"]["source_frame_index"] == 465
    assert samples["first"]["timestamp_seconds"] == 31
    assert samples["last"]["source_frame_index"] == 509
    for sample in samples.values():
        assert 465 <= sample["source_frame_index"] <= 509


def test_denoising_preserves_original_and_declares_transform(tmp_path: Path) -> None:
    source = tmp_path / "original.mp4"
    _encode(source, _add_noise(_scene(45, moving=True), "fine", 7))
    source_bytes = source.read_bytes()
    derivative, record = denoise_mp4(source)
    assert source.read_bytes() == source_bytes
    assert derivative != source
    assert record["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert record["filter"] == EVIDENCE_FILTER
    assert record["playback_rate"] == EVIDENCE_PLAYBACK_RATE
    assert record["changes_simulator_outcome"] is False
    result = probe_mp4(
        derivative,
        evidence_interval={
            "start_action_step": 1,
            "end_action_step": 45,
            "total_action_steps": 45,
        },
    )
    assert result["frame_count"] == 45
    assert result["codec"] == "h264"
    assert result["pixel_format"] == "yuv420p"
    assert result["duration_seconds"] >= 2 * (45 - 1) / 15


@pytest.mark.parametrize("kind", ["fine", "coarse", "flicker"])
def test_denoised_static_render_noise_still_fails(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "noisy-static.mp4"
    _encode(source, _add_noise(_scene(45), kind, 43))
    derivative, _ = denoise_mp4(source)
    with pytest.raises(IsaacArenaError, match="noise-resistant coherent scene motion"):
        probe_mp4(
            derivative,
            evidence_interval={
                "start_action_step": 0,
                "end_action_step": 45,
                "total_action_steps": 45,
            },
        )


def test_short_native_episode_is_slowed_without_adding_frames(tmp_path: Path) -> None:
    source = tmp_path / "short-native-episode.mp4"
    _encode(source, _scene(42, moving=True), rate=50)
    derivative, record = denoise_mp4(source)
    result = probe_mp4(
        derivative,
        evidence_interval={
            "start_action_step": 1,
            "end_action_step": 42,
            "total_action_steps": 42,
        },
    )
    assert result["frame_count"] == 42
    assert result["duration_seconds"] >= 1.6
    assert record["playback_rate"] == 0.5


@pytest.mark.parametrize("grain_size", [20, 40])
@pytest.mark.parametrize("seed", [7, 19, 43])
def test_large_stochastic_blocks_cannot_masquerade_as_object_tracks(
    tmp_path: Path,
    grain_size: int,
    seed: int,
) -> None:
    video = tmp_path / "large-grain-static.mp4"
    _encode(video, _add_noise(_scene(45), "coarse", seed, grain_size))
    with pytest.raises(IsaacArenaError, match="noise-resistant coherent scene motion"):
        probe_mp4(video)


@pytest.mark.parametrize("grain_size", [20, 40])
@pytest.mark.parametrize("seed", [7, 19, 43])
def test_transformed_large_stochastic_blocks_cannot_qualify(
    tmp_path: Path,
    grain_size: int,
    seed: int,
) -> None:
    source = tmp_path / "large-grain-static.mp4"
    _encode(source, _add_noise(_scene(45), "coarse", seed, grain_size))
    derivative, _ = denoise_mp4(source)
    with pytest.raises(IsaacArenaError, match="noise-resistant"):
        probe_mp4(
            derivative,
            evidence_interval={
                "start_action_step": 0,
                "end_action_step": 45,
                "total_action_steps": 45,
            },
            progress_interval={
                "start_action_step": 30,
                "end_action_step": 45,
                "total_action_steps": 45,
            },
            progress_signal="monotonic_structural_change",
            progress_region=_TASK_REGION,
            progress_association_radius_fraction=0.03,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_action_step", -1),
        ("start_action_step", 45),
        ("start_action_step", True),
        ("end_action_step", 46),
        ("total_action_steps", 45.0),
        ("end_action_step", "45"),
    ],
)
def test_invalid_ground_truth_intervals_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    video = tmp_path / "invalid-interval.mp4"
    _encode(video, _scene(45, moving=True))
    interval = {"start_action_step": 1, "end_action_step": 45, "total_action_steps": 45}
    interval[field] = value
    with pytest.raises(IsaacArenaError, match="invalid simulator-ground-truth"):
        probe_mp4(video, evidence_interval=interval)
