"""Unit tests for the NVIDIA Cosmos Evaluator workbench tool.

Covers the algorithm helpers with no network or ffmpeg, the upstream-protocol
question/answer plumbing against a fake Token Factory client, and the run-level
report assembly against a fake storage client.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from npa.workbench.cosmos_evaluator import attribute_verification as av
from npa.workbench.cosmos_evaluator import appearance_fidelity as appearance
from npa.workbench.cosmos_evaluator import hallucination as hal
from npa.workbench.cosmos_evaluator import temporal_consistency as temporal
from npa.workbench.cosmos_evaluator.upstream import CosmosEvaluatorError

APPEARANCE_OPTIONS = {
    "cloth_color": ["blue", "red", "white", "green"],
    "lighting": ["bright daylight", "warm lamp light", "dim evening light"],
}


# ---------------------------------------------------------------------------
# Hallucination check internals
# ---------------------------------------------------------------------------


def _brute_force_edt(dynamic: np.ndarray) -> np.ndarray:
    """Reference squared distance transform, computed pairwise."""

    seeds = np.argwhere(dynamic)
    out = np.empty(dynamic.shape, dtype=np.float64)
    for y in range(dynamic.shape[0]):
        for x in range(dynamic.shape[1]):
            out[y, x] = np.min((seeds[:, 0] - y) ** 2 + (seeds[:, 1] - x) ** 2)
    return out


def test_squared_edt_matches_brute_force() -> None:
    rng = np.random.default_rng(1234)
    dynamic = rng.random((11, 13)) < 0.15
    dynamic[0, 0] = True  # guarantee at least one seed
    assert np.allclose(hal._squared_edt(dynamic), _brute_force_edt(dynamic))


def test_squared_edt_is_all_inf_without_seeds() -> None:
    assert np.isinf(hal._squared_edt(np.zeros((4, 4), dtype=bool))).all()


def test_hallucination_counts_flags_only_distant_motion() -> None:
    original = np.zeros((32, 32), dtype=np.uint8)
    original[10:14, 10:14] = 255
    augmented = np.zeros((32, 32), dtype=np.uint8)
    augmented[10:14, 10:14] = 255  # same motion, close to the original
    augmented[28:31, 28:31] = 255  # invented motion, far away

    hallucinated, total = hal._hallucination_counts(
        original, augmented, dist_tol_px=7.0
    )
    assert total == 25  # 16 shared + 9 invented
    assert hallucinated == 9


def test_hallucination_counts_ignores_a_static_augmented_clip() -> None:
    original = np.zeros((8, 8), dtype=np.uint8)
    original[2:4, 2:4] = 255
    assert hal._hallucination_counts(
        original, np.zeros((8, 8), dtype=np.uint8), 7.0
    ) == (0, 0)


def test_dynamic_mask_thresholds_frame_difference() -> None:
    previous = np.zeros((24, 24), dtype=np.uint8)
    current = np.zeros((24, 24), dtype=np.uint8)
    current[8:16, 8:16] = 200

    mask = hal._dynamic_mask(
        previous, current, grad_thresh=10.0, blur_ksize=1, morph_k=1
    )
    assert mask[12, 12] == 255
    assert mask[0, 0] == 0


def test_gaussian_blur_preserves_a_flat_field() -> None:
    flat = np.full((10, 10), 128, dtype=np.uint8)
    assert np.array_equal(hal._gaussian_blur(flat, 7), flat)


def test_ellipse_element_is_symmetric_and_centered() -> None:
    element = hal._ellipse_element(3)
    assert element.shape == (3, 3)
    assert element[1, 1]
    assert np.array_equal(element, element.T)


def test_numpy_mask_path_matches_opencv_when_opencv_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NumPy kernels stand in for OpenCV's, so they must agree with them."""

    pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
    current = previous.copy()
    current[12:30, 12:30] = 240

    with_opencv = hal._dynamic_mask(
        previous, current, grad_thresh=10.0, blur_ksize=7, morph_k=3
    )
    monkeypatch.setattr(hal, "_cv2", lambda: None)
    with_numpy = hal._dynamic_mask(
        previous, current, grad_thresh=10.0, blur_ksize=7, morph_k=3
    )

    disagreement = np.count_nonzero(with_opencv != with_numpy) / with_opencv.size
    assert disagreement < 0.02, f"masks disagree on {disagreement:.1%} of pixels"


def test_check_hallucination_rejects_a_missing_clip(tmp_path: Path) -> None:
    present = tmp_path / "present.mp4"
    present.write_bytes(b"not really a video")
    with pytest.raises(CosmosEvaluatorError, match="augmented video not found"):
        hal.check_hallucination(
            clip_id="c",
            original_video=present,
            augmented_video=tmp_path / "missing.mp4",
        )


def test_check_hallucination_rejects_an_out_of_range_threshold(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    with pytest.raises(CosmosEvaluatorError, match="threshold"):
        hal.check_hallucination(
            clip_id="c", original_video=clip, augmented_video=clip, threshold=1.5
        )


# ---------------------------------------------------------------------------
# NPA source-relative temporal consistency companion check
# ---------------------------------------------------------------------------


def _frame_stream(values: list[np.ndarray]):
    yield from values


def _run_temporal_with_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: list[np.ndarray],
    augmented: list[np.ndarray],
    **kwargs: Any,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    original = tmp_path / "original.mp4"
    variant = tmp_path / "variant.mp4"
    original.write_bytes(b"video")
    variant.write_bytes(b"video")
    monkeypatch.setattr(temporal, "_probe_size", lambda path: source[0].shape)
    streams = iter([source, augmented])
    monkeypatch.setattr(
        temporal,
        "_iter_gray_frames",
        lambda path, height, width: _frame_stream(next(streams)),
    )
    return temporal.check_temporal_consistency(
        clip_id="clip-0",
        original_video=original,
        augmented_video=variant,
        **kwargs,
    )


def test_temporal_consistency_accepts_source_matching_motion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [np.full((8, 8), value, dtype=np.uint8) for value in (0, 8, 19, 33)]
    result = _run_temporal_with_frames(
        tmp_path, monkeypatch, frames, [f.copy() for f in frames]
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.frame_counts_match is True
    assert len(result.regions) == 5


def test_temporal_consistency_rejects_local_excess_acceleration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.zeros((8, 8), dtype=np.uint8) for _ in range(5)]
    augmented = [frame.copy() for frame in source]
    for index, frame in enumerate(augmented):
        frame[:4, :4] = 80 if index % 2 else 0
    result = _run_temporal_with_frames(tmp_path, monkeypatch, source, augmented)
    assert result.passed is False
    assert result.score < 0.75
    assert (
        next(region for region in result.regions if region.region_id == "tile-0").passed
        is False
    )


def test_temporal_consistency_rejects_a_frame_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.full((4, 4), value, dtype=np.uint8) for value in (0, 2, 4, 6)]
    augmented = [frame.copy() for frame in source[:3]]
    result = _run_temporal_with_frames(tmp_path, monkeypatch, source, augmented)
    assert result.frame_counts_match is False
    assert result.passed is False


def test_temporal_regions_validate_normalized_bounds() -> None:
    assert temporal.parse_regions("[[0.1, 0.2, 0.8, 0.9]]")[0][1] == (
        0.1,
        0.2,
        0.8,
        0.9,
    )
    with pytest.raises(CosmosEvaluatorError, match="bounds"):
        temporal.parse_regions("[[0, 0, 2, 1]]")


def test_temporal_consistency_rejects_a_missing_clip(tmp_path: Path) -> None:
    present = tmp_path / "present.mp4"
    present.write_bytes(b"video")
    with pytest.raises(CosmosEvaluatorError, match="original video not found"):
        temporal.check_temporal_consistency(
            clip_id="c",
            original_video=tmp_path / "missing.mp4",
            augmented_video=present,
        )


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_temporal_consistency_rejects_an_out_of_range_threshold(
    tmp_path: Path, threshold: float
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    with pytest.raises(CosmosEvaluatorError, match="threshold"):
        temporal.check_temporal_consistency(
            clip_id="c", original_video=clip, augmented_video=clip, threshold=threshold
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"noise_floor": 0.0}, "noise floor"),
        ({"blur_ksize": 0}, "blur kernel"),
        ({"blur_ksize": 4}, "blur kernel"),
    ],
)
def test_temporal_consistency_validates_calibration_parameters(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    with pytest.raises(CosmosEvaluatorError, match=message):
        temporal.check_temporal_consistency(
            clip_id="c", original_video=clip, augmented_video=clip, **kwargs
        )


def test_temporal_consistency_requires_three_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [np.zeros((4, 4), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(CosmosEvaluatorError, match="at least three"):
        _run_temporal_with_frames(
            tmp_path, monkeypatch, frames, [f.copy() for f in frames]
        )


def test_temporal_consistency_rejects_collapsed_source_motion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.full((8, 8), value, dtype=np.uint8) for value in (40, 70, 100, 70, 40)]
    frozen = [np.full((8, 8), 70, dtype=np.uint8) for _ in source]
    result = _run_temporal_with_frames(
        tmp_path,
        monkeypatch,
        source,
        frozen,
        blur_ksize=1,
        noise_floor=0.25,
    )
    assert result.passed is False
    assert result.score < 0.8
    assert result.regions[0].source_mean_acceleration > 0
    assert result.regions[0].augmented_mean_acceleration == 0


def test_same_artifact_scores_equally_across_source_motion_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static = [np.full((8, 8), 80, dtype=np.uint8) for _ in range(5)]
    moving = [np.full((8, 8), value, dtype=np.uint8) for value in (40, 70, 100, 70, 40)]
    artifact = (0, 5, 0, 5, 0)
    static_augmented = [
        frame + value for frame, value in zip(static, artifact, strict=True)
    ]
    moving_augmented = [
        frame + value for frame, value in zip(moving, artifact, strict=True)
    ]

    static_result = _run_temporal_with_frames(
        tmp_path / "static",
        monkeypatch,
        static,
        static_augmented,
        blur_ksize=1,
        noise_floor=0.25,
    )
    moving_result = _run_temporal_with_frames(
        tmp_path / "moving",
        monkeypatch,
        moving,
        moving_augmented,
        blur_ksize=1,
        noise_floor=0.25,
    )
    assert static_result.score == moving_result.score
    assert static_result.regions[0].residual_mean_acceleration == (
        moving_result.regions[0].residual_mean_acceleration
    )


def test_temporal_prefilter_suppresses_isolated_pixel_noise() -> None:
    impulse = np.zeros((17, 17), dtype=np.uint8)
    impulse[8, 8] = 255
    filtered = temporal._prefilter(impulse, 7)
    assert filtered.max() < impulse.max()
    assert np.count_nonzero(filtered) > 1


def test_temporal_consistency_uses_source_geometry_for_both_decoders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original.mp4"
    augmented = tmp_path / "augmented.mp4"
    original.write_bytes(b"video")
    augmented.write_bytes(b"video")
    calls: list[tuple[str, int, int]] = []
    frames = [np.zeros((6, 10), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(temporal, "_probe_size", lambda path: (6, 10))

    def decode(path: Path, height: int, width: int):
        calls.append((path.name, height, width))
        yield from [frame.copy() for frame in frames]

    monkeypatch.setattr(temporal, "_iter_gray_frames", decode)
    temporal.check_temporal_consistency(
        clip_id="c", original_video=original, augmented_video=augmented
    )
    assert calls == [("original.mp4", 6, 10), ("augmented.mp4", 6, 10)]


def test_temporal_regions_accept_labels_and_positional_entries() -> None:
    parsed = temporal.parse_regions(
        '[{"id":"subject","bounds":[0,0,0.5,1]},[0.5,0,1,1]]'
    )
    assert parsed == [
        ("subject", (0.0, 0.0, 0.5, 1.0)),
        ("region-1", (0.5, 0.0, 1.0, 1.0)),
    ]
    assert temporal.parse_regions("") == list(temporal.DEFAULT_REGIONS)


@pytest.mark.parametrize(
    "payload",
    ["{", "{}", "[]", "[[0, 0, 1]]", '[[0, 0, "x", 1]]', "[[0.5,0,0.5,1]]"],
)
def test_temporal_regions_reject_malformed_payloads(payload: str) -> None:
    with pytest.raises(CosmosEvaluatorError):
        temporal.parse_regions(payload)


def test_temporal_consistency_honors_caller_regions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.zeros((8, 8), dtype=np.uint8) for _ in range(5)]
    augmented = [frame.copy() for frame in source]
    for index, frame in enumerate(augmented):
        frame[:4, :4] = 80 if index % 2 else 0
    result = _run_temporal_with_frames(
        tmp_path,
        monkeypatch,
        source,
        augmented,
        regions='[{"id":"clean","bounds":[0.5,0.5,1,1]}]',
        blur_ksize=1,
    )
    assert result.passed is True
    assert [region.region_id for region in result.regions] == ["clean"]


def test_temporal_consistency_decodes_real_encoded_video(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip(
            "ffmpeg and ffprobe are required for the encoded-video integration test"
        )
    original = tmp_path / "original.mp4"
    augmented = tmp_path / "augmented.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=6:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(original),
        ],
        check=True,
    )
    shutil.copyfile(original, augmented)
    result = temporal.check_temporal_consistency(
        clip_id="encoded", original_video=original, augmented_video=augmented
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.total_frames >= 3


# ---------------------------------------------------------------------------
# NPA source-relative protected-appearance fidelity companion check
# ---------------------------------------------------------------------------


def _run_appearance_with_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: list[np.ndarray],
    augmented: list[np.ndarray],
    **kwargs: Any,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    original = tmp_path / "original.mp4"
    variant = tmp_path / "variant.mp4"
    original.write_bytes(b"video")
    variant.write_bytes(b"video")
    monkeypatch.setattr(appearance, "_probe_size", lambda path: source[0].shape[:2])
    streams = iter([source, augmented])
    monkeypatch.setattr(
        appearance,
        "_iter_rgb_frames",
        lambda path, height, width: _frame_stream(next(streams)),
    )
    return appearance.check_appearance_fidelity(
        clip_id="clip-0",
        original_video=original,
        augmented_video=variant,
        blur_ksize=1,
        **kwargs,
    )


def test_appearance_fidelity_accepts_matching_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [np.full((8, 10, 3), 100 + index, dtype=np.uint8) for index in range(4)]
    result = _run_appearance_with_frames(
        tmp_path, monkeypatch, frames, [frame.copy() for frame in frames]
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.frame_counts_match is True
    assert len(result.regions) == 5


def test_appearance_fidelity_accepts_bounded_global_photometric_shift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.full((8, 10, 3), 100, dtype=np.uint8) for _ in range(4)]
    augmented = [frame.copy() for frame in source]
    for frame in augmented:
        frame[:, :, 0] += 6
        frame[:, :, 1] += 3
        frame[:, :, 2] += 1
    result = _run_appearance_with_frames(tmp_path, monkeypatch, source, augmented)
    assert result.passed is True
    assert result.score == 1.0


def test_appearance_fidelity_rejects_excessive_global_colour_cast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.full((8, 10, 3), 100, dtype=np.uint8) for _ in range(4)]
    augmented = [frame.copy() for frame in source]
    for frame in augmented:
        frame[:, :, 0] = 190
        frame[:, :, 1] = 60
        frame[:, :, 2] = 50
    result = _run_appearance_with_frames(tmp_path, monkeypatch, source, augmented)
    assert result.passed is False
    assert result.score < 0.8
    full = next(region for region in result.regions if region.region_id == "full-frame")
    assert full.chroma_delta_p95 > result.global_chroma_tolerance


def test_appearance_fidelity_rejects_localized_material_recolouring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.full((8, 10, 3), 100, dtype=np.uint8) for _ in range(4)]
    augmented = [frame.copy() for frame in source]
    for frame in augmented:
        frame[4:, :5, 0] = 200
        frame[4:, :5, 1:] = 40
    result = _run_appearance_with_frames(tmp_path, monkeypatch, source, augmented)
    assert result.passed is False
    affected = next(region for region in result.regions if region.region_id == "tile-2")
    assert affected.local_chroma_residual_p95 > result.local_chroma_tolerance


def test_appearance_fidelity_rejects_chroma_shift_instability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.full((8, 10, 3), 100, dtype=np.uint8) for _ in range(5)]
    augmented = [frame.copy() for frame in source]
    for index, frame in enumerate(augmented):
        if index % 2:
            frame[:, :, 0] = 120
            frame[:, :, 2] = 80
    result = _run_appearance_with_frames(tmp_path, monkeypatch, source, augmented)
    assert result.passed is False
    assert any(
        region.chroma_instability_p95
        > result.chroma_instability_tolerance
        for region in result.regions
    )


def test_appearance_fidelity_rejects_frame_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [np.full((4, 4, 3), 100, dtype=np.uint8) for _ in range(4)]
    augmented = [frame.copy() for frame in source[:3]]
    result = _run_appearance_with_frames(tmp_path, monkeypatch, source, augmented)
    assert result.frame_counts_match is False
    assert result.passed is False


def test_appearance_regions_accept_labels_and_validate_bounds() -> None:
    parsed = appearance.parse_regions(
        '[{"id":"protected","bounds":[0.1,0.2,0.8,0.9]}]'
    )
    assert parsed == [("protected", (0.1, 0.2, 0.8, 0.9))]
    assert appearance.parse_regions("") == list(appearance.DEFAULT_REGIONS)
    with pytest.raises(CosmosEvaluatorError, match="bounds"):
        appearance.parse_regions("[[0,0,2,1]]")


def test_appearance_fidelity_decodes_real_encoded_video(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip(
            "ffmpeg and ffprobe are required for the encoded-video integration test"
        )
    original = tmp_path / "original.mp4"
    augmented = tmp_path / "augmented.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=6:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(original),
        ],
        check=True,
    )
    shutil.copyfile(original, augmented)
    result = appearance.check_appearance_fidelity(
        clip_id="encoded", original_video=original, augmented_video=augmented
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.total_frames >= 1


# ---------------------------------------------------------------------------
# Attribute verification: upstream's question / answer protocol
# ---------------------------------------------------------------------------


class FakeTokenFactory:
    """Records requests and replays scripted replies."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def chat_completion_text(self, **kwargs: Any) -> str:
        self.requests.append(kwargs)
        if not self.replies:
            raise AssertionError("FakeTokenFactory ran out of scripted replies")
        return self.replies.pop(0)


def _question_json(variable: str, value: str, other: str, *, correct: str = "A") -> str:
    options = {"A": value, "B": other} if correct == "A" else {"A": other, "B": value}
    return json.dumps(
        {
            "variable": variable,
            "value": value,
            "question": f"What is the {variable}?",
            "options": options,
            "correct_answer": correct,
        }
    )


def test_parse_question_response_reads_a_fenced_block() -> None:
    payload = _question_json("cloth_color", "blue", "red")
    parsed = av.parse_question_response(
        f"Sure, here you go:\n```json\n{payload}\n```\n"
    )
    assert parsed["variable"] == "cloth_color"


def test_parse_question_response_drops_a_reasoning_block() -> None:
    payload = _question_json("cloth_color", "blue", "red")
    parsed = av.parse_question_response(f"<think>weighing options {{</think>{payload}")
    assert parsed["options"]["A"] == "blue"


def test_parse_question_response_unwraps_a_single_item_list() -> None:
    payload = json.loads(
        _question_json("lighting", "warm lamp light", "bright daylight")
    )
    assert av.parse_question_response(json.dumps([payload]))["variable"] == "lighting"


def test_parse_question_response_rejects_unparseable_output() -> None:
    with pytest.raises(CosmosEvaluatorError, match="could not parse"):
        av.parse_question_response("I would rather not answer.")


def test_normalize_question_repairs_a_wrong_answer_key() -> None:
    raw = json.loads(_question_json("cloth_color", "blue", "red", correct="B"))
    # The generator labelled B as correct while B is the *other* value.
    normalized = av.normalize_question(
        raw,
        variable="cloth_color",
        value="blue",
        options=APPEARANCE_OPTIONS["cloth_color"],
    )
    assert normalized["options"][normalized["correct_answer"]] == "blue"


def test_normalize_question_rejects_options_without_the_requested_value() -> None:
    raw = {
        "variable": "cloth_color",
        "value": "blue",
        "question": "What colour is the cloth?",
        "options": {"A": "red", "B": "green"},
        "correct_answer": "A",
    }
    with pytest.raises(CosmosEvaluatorError, match="omits the requested value"):
        av.normalize_question(
            raw, variable="cloth_color", value="blue", options=["red", "green"]
        )


def test_normalize_question_rejects_a_single_option() -> None:
    raw = {
        "variable": "cloth_color",
        "value": "blue",
        "question": "What colour?",
        "options": {"A": "blue"},
        "correct_answer": "A",
    }
    with pytest.raises(CosmosEvaluatorError, match="options"):
        av.normalize_question(
            raw, variable="cloth_color", value="blue", options=["blue"]
        )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("A", "A"),
        ("The answer is C.", "C"),
        ("b", "B"),
        ("no idea", "UNKNOWN"),
        ("", "UNKNOWN"),
    ],
)
def test_parse_answer_letter(reply: str, expected: str) -> None:
    assert av.parse_answer_letter(reply) == expected


def test_parse_answer_letter_rejects_a_letter_the_question_never_offered() -> None:
    """A live VLM does answer "D" to a three-option question; that is not an answer."""

    assert av.parse_answer_letter("D", offered=["A", "B", "C"]) == "UNKNOWN"
    assert av.parse_answer_letter("D", offered=["A", "B", "C", "D"]) == "D"


def test_parse_answer_letter_skips_past_an_unoffered_letter() -> None:
    assert av.parse_answer_letter("Not D, it is B.", offered=["A", "B"]) == "B"


def test_answer_question_constrains_the_answer_to_the_offered_options() -> None:
    client = FakeTokenFactory(["D"])
    answer = av.answer_question(
        client=client,
        question="What colour?",
        options={"A": "red", "B": "blue", "C": "green"},
        data_url="data:image/png;base64,AAA=",
        model="vlm",
    )
    assert answer == "UNKNOWN"


def test_generate_question_uses_upstream_guided_json_schema() -> None:
    client = FakeTokenFactory([_question_json("cloth_color", "blue", "red")])
    av.generate_question(
        client=client,
        variable="cloth_color",
        value="blue",
        options=APPEARANCE_OPTIONS["cloth_color"],
        model="llm",
    )
    assert client.requests[0]["response_format"] is av.QUESTION_SCHEMA
    prompt = client.requests[0]["messages"][1]["content"]
    assert "blue" in prompt and "green" in prompt


def test_generate_question_retries_without_structured_output() -> None:
    class RejectsSchema(FakeTokenFactory):
        def chat_completion_text(self, **kwargs: Any) -> str:
            if "response_format" in kwargs:
                self.requests.append(kwargs)
                raise RuntimeError("response_format is not supported")
            return super().chat_completion_text(**kwargs)

    client = RejectsSchema([_question_json("cloth_color", "blue", "red")])
    question = av.generate_question(
        client=client,
        variable="cloth_color",
        value="blue",
        options=APPEARANCE_OPTIONS["cloth_color"],
        model="llm",
    )
    assert question["correct_answer"] == "A"
    assert len(client.requests) == 2


def test_verify_attributes_scores_a_matching_answer(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    client = FakeTokenFactory(
        [
            _question_json("cloth_color", "blue", "red"),
            "A",
            _question_json("lighting", "warm lamp light", "bright daylight"),
            "B",  # wrong: B is the other value
        ]
    )
    result = av.verify_attributes(
        clip_id="clip-0",
        frame=frame,
        selected_variables={"cloth_color": "blue", "lighting": "warm lamp light"},
        variable_options=APPEARANCE_OPTIONS,
        client=client,
    )
    assert result.total_checks == 2
    assert result.passed_checks == 1
    assert result.score == 0.5
    assert result.passed is False
    assert [check.variable for check in result.checks] == ["cloth_color", "lighting"]


def test_verify_attributes_records_a_failing_check_without_dropping_the_batch(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    class FailsFirstQuestion(FakeTokenFactory):
        def chat_completion_text(self, **kwargs: Any) -> str:
            if not self.requests and "response_format" in kwargs:
                self.requests.append(kwargs)
                raise RuntimeError("endpoint exploded")
            return super().chat_completion_text(**kwargs)

    client = FailsFirstQuestion(
        [_question_json("lighting", "warm lamp light", "bright daylight"), "A"]
    )
    result = av.verify_attributes(
        clip_id="clip-0",
        frame=frame,
        selected_variables={"cloth_color": "blue", "lighting": "warm lamp light"},
        variable_options=APPEARANCE_OPTIONS,
        client=client,
    )
    assert result.total_checks == 2
    assert result.checks[0].error
    assert result.checks[1].passed
    assert result.score == 0.5


class _RefusesGuidedJson(FakeTokenFactory):
    """An endpoint that cannot honour ``response_format`` but answers plain calls."""

    def chat_completion_text(self, **kwargs: Any) -> str:
        if "response_format" in kwargs:
            raise RuntimeError("400: response_format is not supported by this model")
        return super().chat_completion_text(**kwargs)


class _RateLimited(FakeTokenFactory):
    """An endpoint under load. Retrying the same call just doubles the load."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.calls = 0

    def chat_completion_text(self, **kwargs: Any) -> str:
        self.calls += 1
        raise RuntimeError("429: rate limit exceeded")


def test_an_endpoint_without_guided_json_is_retried_unstructured() -> None:
    client = _RefusesGuidedJson([_question_json("cloth_color", "blue", "red")])
    question = av.generate_question(
        client=client,
        variable="cloth_color",
        value="blue",
        options=["blue", "red"],
        model="m",
    )
    assert question["correct_answer"]


def test_a_rate_limit_is_not_retried_as_if_it_were_a_schema_problem() -> None:
    """Retrying a 429 doubles the load and hides the cause behind a JSON message."""

    client = _RateLimited([])
    with pytest.raises(RuntimeError, match="rate limit"):
        av.generate_question(
            client=client,
            variable="cloth_color",
            value="blue",
            options=["blue"],
            model="m",
        )
    assert client.calls == 1


def test_verify_attributes_requires_exactly_one_media_source(tmp_path: Path) -> None:
    with pytest.raises(CosmosEvaluatorError, match="exactly one"):
        av.verify_attributes(
            clip_id="c",
            selected_variables={"cloth_color": "blue"},
            client=FakeTokenFactory([]),
        )


def test_verify_attributes_requires_a_variable(tmp_path: Path) -> None:
    with pytest.raises(CosmosEvaluatorError, match="at least one selected variable"):
        av.verify_attributes(
            clip_id="c", frame=tmp_path / "f.png", selected_variables={}
        )


def test_format_question_lists_options_in_letter_order() -> None:
    text = av.format_question("What colour?", {"B": "red", "A": "blue"})
    assert text.splitlines()[1:3] == ["A) blue", "B) red"]


# ---------------------------------------------------------------------------
# Run-level report assembly
# ---------------------------------------------------------------------------


def _write_variant(
    root: Path, clip: str, variables: dict[str, str], *, conditioned: bool
) -> None:
    variant = root / clip
    variant.mkdir(parents=True)
    (variant / "frame-000.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (variant / "metadata.json").write_text(
        json.dumps({"variables": variables, "input_conditioned": conditioned}),
        encoding="utf-8",
    )


def test_evaluate_run_grades_every_local_variant(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    _write_variant(
        augment,
        "clip-a",
        {"cloth_color": "blue", "prompt": "ignored"},
        conditioned=False,
    )
    _write_variant(augment, "clip-b", {"cloth_color": "red"}, conditioned=False)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "manifest.json").write_text(
        json.dumps({"variables": APPEARANCE_OPTIONS}), encoding="utf-8"
    )

    client = FakeTokenFactory(
        [
            _question_json("cloth_color", "blue", "red"),
            "A",
            _question_json("cloth_color", "red", "blue"),
            "A",
        ]
    )
    result = evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        configs_uri=str(configs),
        client=client,
        storage=object(),
    )
    assert result.clip_count == 2
    assert result.score == 1.0
    assert result.passed is True
    assert [clip.clip_id for clip in result.clips] == ["clip-a", "clip-b"]
    # `prompt` is an instruction, not a visual attribute, so it is never asked about.
    assert "prompt" not in result.clips[0].variables
    # Without a source clip the hallucination check is skipped, and says so.
    assert any("hallucination" in reason for reason in result.clips[0].skipped)


def test_evaluate_run_rejects_an_empty_augment_prefix(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    empty = tmp_path / "cosmos_augmented"
    empty.mkdir()
    with pytest.raises(CosmosEvaluatorError, match="no augmented variant directories"):
        evaluate_run(
            augment_uri=str(empty), output_uri=str(tmp_path / "out"), storage=object()
        )


def test_combine_scores_ignores_hallucination_for_unconditioned_variants() -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _combine_scores

    attribute = av.AttributeVerificationResult(
        clip_id="c",
        passed=True,
        total_checks=2,
        passed_checks=2,
        failed_checks=0,
        score=1.0,
        question_model="llm",
        vlm_model="vlm",
    )
    hallucinated = hal.HallucinationResult(
        clip_id="c",
        passed=False,
        threshold=0.682,
        score=0.2,
        total_frames=10,
        total_hallucinated_dynamic_pixels=80,
        total_augmented_dynamic_pixels=100,
    )
    unconditioned = _combine_scores(
        attribute_result=attribute,
        hallucination_result=hallucinated,
        input_conditioned=False,
        hallucination_weight=0.5,
    )
    assert unconditioned == (1.0, True)

    conditioned = _combine_scores(
        attribute_result=attribute,
        hallucination_result=hallucinated,
        input_conditioned=True,
        hallucination_weight=0.5,
    )
    assert conditioned == (0.6, False)


def test_combine_scores_treats_temporal_consistency_as_a_hard_check() -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _combine_scores

    attribute = av.AttributeVerificationResult(
        clip_id="c",
        passed=True,
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        score=1.0,
        question_model="llm",
        vlm_model="vlm",
    )
    hallucination = hal.HallucinationResult(
        clip_id="c",
        passed=True,
        threshold=0.75,
        score=0.95,
        total_frames=10,
        total_hallucinated_dynamic_pixels=1,
        total_augmented_dynamic_pixels=20,
    )
    temporal_result = temporal.TemporalConsistencyResult(
        clip_id="c",
        passed=False,
        threshold=0.75,
        score=0.6,
        total_frames=10,
        frame_counts_match=True,
    )
    assert _combine_scores(
        attribute_result=attribute,
        hallucination_result=hallucination,
        input_conditioned=True,
        hallucination_weight=0.5,
        temporal_result=temporal_result,
        temporal_required=True,
    ) == (0.6, False)


def test_combine_scores_keeps_every_attribute_check_hard() -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _combine_scores

    attribute = av.AttributeVerificationResult(
        clip_id="c",
        passed=False,
        total_checks=4,
        passed_checks=2,
        failed_checks=2,
        score=0.5,
        question_model="llm",
        vlm_model="vlm",
    )
    hallucination = hal.HallucinationResult(
        clip_id="c",
        passed=True,
        threshold=0.75,
        score=1.0,
        total_frames=10,
        total_hallucinated_dynamic_pixels=0,
        total_augmented_dynamic_pixels=20,
    )
    temporal_result = temporal.TemporalConsistencyResult(
        clip_id="c",
        passed=True,
        threshold=0.8,
        score=1.0,
        total_frames=10,
        frame_counts_match=True,
    )
    assert _combine_scores(
        attribute_result=attribute,
        hallucination_result=hallucination,
        input_conditioned=True,
        hallucination_weight=0.5,
        temporal_result=temporal_result,
        temporal_required=True,
        score_threshold=0.75,
    ) == (0.75, False)


def test_combine_scores_keeps_temporal_advisory_by_default() -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _combine_scores

    attribute = av.AttributeVerificationResult(
        clip_id="c",
        passed=True,
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        score=1.0,
        question_model="llm",
        vlm_model="vlm",
    )
    hallucination = hal.HallucinationResult(
        clip_id="c",
        passed=True,
        threshold=0.75,
        score=0.9,
        total_frames=10,
        total_hallucinated_dynamic_pixels=1,
        total_augmented_dynamic_pixels=20,
    )
    diagnostic = temporal.TemporalConsistencyResult(
        clip_id="c",
        passed=False,
        threshold=0.8,
        score=0.1,
        total_frames=10,
        frame_counts_match=True,
    )
    assert _combine_scores(
        attribute_result=attribute,
        hallucination_result=hallucination,
        input_conditioned=True,
        hallucination_weight=0.5,
        temporal_result=diagnostic,
        temporal_required=False,
        score_threshold=0.75,
    ) == (0.95, True)


def test_combine_scores_treats_required_appearance_as_a_hard_check() -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _combine_scores

    attribute = av.AttributeVerificationResult(
        clip_id="c",
        passed=True,
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        score=1.0,
        question_model="llm",
        vlm_model="vlm",
    )
    hallucination = hal.HallucinationResult(
        clip_id="c",
        passed=True,
        threshold=0.75,
        score=0.95,
        total_frames=10,
        total_hallucinated_dynamic_pixels=1,
        total_augmented_dynamic_pixels=20,
    )
    appearance_result = appearance.AppearanceFidelityResult(
        clip_id="c",
        passed=False,
        threshold=0.8,
        score=0.2,
        total_frames=10,
        frame_counts_match=True,
    )
    assert _combine_scores(
        attribute_result=attribute,
        hallucination_result=hallucination,
        input_conditioned=True,
        hallucination_weight=0.5,
        appearance_result=appearance_result,
        appearance_required=True,
    ) == (0.2, False)


def test_combine_scores_is_zero_when_both_checks_are_missing() -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _combine_scores

    assert _combine_scores(
        attribute_result=None,
        hallucination_result=None,
        input_conditioned=True,
        hallucination_weight=0.5,
    ) == (0.0, False)


def test_conditioned_clip_cannot_pass_on_attributes_without_hallucination() -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _combine_scores

    attribute = av.AttributeVerificationResult(
        clip_id="conditioned",
        passed=True,
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        score=1.0,
        question_model="llm",
        vlm_model="vlm",
    )

    assert _combine_scores(
        attribute_result=attribute,
        hallucination_result=None,
        input_conditioned=True,
        hallucination_weight=0.5,
    ) == (0.0, False)


def test_report_uri_for_appends_the_result_filename() -> None:
    from npa.workbench.cosmos_evaluator import RESULT_FILENAME, report_uri_for

    assert report_uri_for("s3://b/run/grade/") == f"s3://b/run/grade/{RESULT_FILENAME}"
    assert (
        report_uri_for("s3://b/run/grade/custom.json") == "s3://b/run/grade/custom.json"
    )


def test_write_report_round_trips_locally(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator.evaluate import write_report

    written = write_report({"score": 0.75}, result_uri=str(tmp_path / "grade"))
    assert json.loads(Path(written).read_text())["score"] == 0.75


def test_local_run_needs_no_object_storage_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local --augment-uri must not require an S3 endpoint to be configured."""
    from npa.workbench.cosmos_evaluator import evaluate_run

    for name in ("AWS_ENDPOINT_URL", "NEBIUS_S3_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)

    augment = tmp_path / "cosmos_augmented"
    _write_variant(augment, "clip-a", {"cloth_color": "blue"}, conditioned=False)
    client = FakeTokenFactory([_question_json("cloth_color", "blue", "red"), "A"])

    result = evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        client=client,
    )
    assert result.clip_count == 1
    written = write_report_helper(result, tmp_path)
    assert json.loads(Path(written).read_text())["clip_count"] == 1


class _BrokenStore:
    """Object storage that answers a listing and then stops answering.

    Mirrors an expired credential or endpoint failure part-way through a run: the
    variants exist, so every subsequent clip would skip for a reason that has nothing
    to do with the clips.
    """

    def __init__(self, clips: list[str]) -> None:
        contents = [{"Key": f"cosmos_augmented/{clip}/metadata.json"} for clip in clips]
        self.s3 = SimpleNamespace(
            list_objects_v2=lambda **kwargs: {
                "Contents": contents,
                "IsTruncated": False,
            }
        )

    def download_path(self, uri: str, dest: str) -> str:
        from botocore.exceptions import EndpointConnectionError

        raise EndpointConnectionError(endpoint_url="https://storage.invalid")


def test_a_storage_outage_is_reported_degraded_not_as_a_batch_of_zeros(
    tmp_path: Path,
) -> None:
    """An outage and a genuinely bad batch must not produce the same report.

    Without this, every clip skips, the mean is 0.0, and the run reads as
    ``completed`` — a quality gate acting on that cannot tell the two apart.
    """

    from npa.workbench.cosmos_evaluator import evaluate_run

    result = evaluate_run(
        augment_uri="s3://bucket/cosmos_augmented/",
        output_uri="s3://bucket/grade/",
        storage=_BrokenStore(["clip-a", "clip-b"]),
    )
    assert result.status == "degraded"
    assert result.passed is False
    assert any("could not read" in warning for warning in result.warnings), (
        result.warnings
    )


def test_an_absent_variant_is_a_skip_not_an_outage(tmp_path: Path) -> None:
    """The mirror case: nothing to read is a legitimate zero, and stays completed."""

    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    (augment / "clip-a").mkdir(parents=True)  # a variant dir with no artifacts in it

    result = evaluate_run(augment_uri=str(augment), output_uri=str(tmp_path / "grade"))
    assert result.status == "completed"
    assert result.score == 0.0
    assert result.clips[0].skipped


def test_one_failed_variant_rejects_a_batch_even_when_the_mean_clears_threshold(
    tmp_path: Path,
) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    for index in range(4):
        _write_variant(
            augment,
            f"clip-{index}",
            {"cloth_color": "blue"},
            conditioned=False,
        )
    replies: list[str] = []
    for index in range(4):
        replies.extend(
            [_question_json("cloth_color", "blue", "red"), "A" if index < 3 else "B"]
        )
    result = evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        threshold=0.5,
        client=FakeTokenFactory(replies),
        storage=object(),
    )
    assert result.score == 0.75
    assert result.passed_clips == 3
    assert result.passed is False
    assert result.batch_policy == "all-variants"


def test_input_conditioned_variant_without_a_source_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import npa.workbench.cosmos_evaluator.evaluate as evaluator

    augment = tmp_path / "cosmos_augmented"
    _write_variant(augment, "clip-a", {"cloth_color": "blue"}, conditioned=True)
    (augment / "clip-a" / "augmented_video.mp4").write_bytes(b"video")
    attribute = av.AttributeVerificationResult(
        clip_id="clip-a",
        passed=True,
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        score=1.0,
        question_model="llm",
        vlm_model="vlm",
    )
    monkeypatch.setattr(evaluator, "verify_attributes", lambda **kwargs: attribute)
    result = evaluator.evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        storage=object(),
    )
    assert result.status == "completed"
    assert result.passed is False
    assert result.clips[0].score == 0.0
    assert any("source clip" in reason for reason in result.clips[0].skipped)


def test_required_appearance_failure_rejects_an_input_conditioned_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import npa.workbench.cosmos_evaluator.evaluate as evaluator

    augment = tmp_path / "cosmos_augmented"
    _write_variant(augment, "clip-a", {"lighting": "bright"}, conditioned=True)
    (augment / "clip-a" / "augmented_video.mp4").write_bytes(b"video")
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "source.mp4").write_bytes(b"video")
    attribute = av.AttributeVerificationResult(
        clip_id="clip-a",
        passed=True,
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        score=1.0,
        question_model="llm",
        vlm_model="vlm",
    )
    hallucination = hal.HallucinationResult(
        clip_id="clip-a",
        passed=True,
        threshold=0.75,
        score=1.0,
        total_frames=10,
        total_hallucinated_dynamic_pixels=0,
        total_augmented_dynamic_pixels=10,
    )
    temporal_result = temporal.TemporalConsistencyResult(
        clip_id="clip-a",
        passed=True,
        threshold=0.8,
        score=1.0,
        total_frames=10,
        frame_counts_match=True,
    )
    appearance_result = appearance.AppearanceFidelityResult(
        clip_id="clip-a",
        passed=False,
        threshold=0.8,
        score=0.25,
        total_frames=10,
        frame_counts_match=True,
    )
    monkeypatch.setattr(evaluator, "verify_attributes", lambda **kwargs: attribute)
    monkeypatch.setattr(
        evaluator, "check_hallucination", lambda **kwargs: hallucination
    )
    monkeypatch.setattr(
        evaluator, "check_temporal_consistency", lambda **kwargs: temporal_result
    )
    monkeypatch.setattr(
        evaluator, "check_appearance_fidelity", lambda **kwargs: appearance_result
    )
    result = evaluator.evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        input_uri=str(inputs),
        appearance_mode="required",
        storage=object(),
    )
    assert result.status == "completed"
    assert result.passed is False
    assert result.clips[0].score == 0.25
    assert result.clips[0].appearance_enforced is True
    assert result.clips[0].appearance_fidelity is not None


def test_temporal_check_stays_skipped_for_unconditioned_variants(
    tmp_path: Path,
) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    _write_variant(augment, "clip-a", {"cloth_color": "blue"}, conditioned=False)
    result = evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        client=FakeTokenFactory([_question_json("cloth_color", "blue", "red"), "A"]),
        storage=object(),
    )
    assert result.clips[0].temporal_consistency is None
    assert result.clips[0].temporal_enforced is False
    assert any("input-conditioned" in reason for reason in result.clips[0].skipped)


def test_appearance_check_stays_skipped_for_unconditioned_variants(
    tmp_path: Path,
) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    _write_variant(augment, "clip-a", {"cloth_color": "blue"}, conditioned=False)
    result = evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        client=FakeTokenFactory([_question_json("cloth_color", "blue", "red"), "A"]),
        storage=object(),
    )
    assert result.clips[0].appearance_fidelity is None
    assert result.clips[0].appearance_enforced is False
    assert any("appearance fidelity" in reason for reason in result.clips[0].skipped)


def test_evaluate_run_rejects_invalid_temporal_configuration(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    with pytest.raises(CosmosEvaluatorError, match="temporal-threshold"):
        evaluate_run(
            augment_uri=str(tmp_path / "augment"),
            output_uri=str(tmp_path / "grade"),
            temporal_threshold=1.1,
            storage=object(),
        )


def test_evaluate_run_rejects_invalid_appearance_configuration(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    with pytest.raises(CosmosEvaluatorError, match="appearance-threshold"):
        evaluate_run(
            augment_uri=str(tmp_path / "augment"),
            output_uri=str(tmp_path / "grade"),
            appearance_threshold=1.1,
            storage=object(),
        )


def test_variant_source_is_matched_from_its_recorded_conditioned_input(
    tmp_path: Path,
) -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _select_source_clip

    sources = [tmp_path / "source-a.mp4", tmp_path / "source-b.mov"]
    warnings: list[str] = []
    selected = _select_source_clip(
        clip_id="variant-b",
        metadata={"conditioned_input": "nested/source-b.mov"},
        source_clips=sources,
        warnings=warnings,
    )
    assert selected == sources[1]
    assert warnings == []


def test_multiple_sources_never_fall_back_to_the_first_key(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _select_source_clip

    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    warnings: list[str] = []
    selected = _select_source_clip(
        clip_id="variant",
        metadata={},
        source_clips=sources,
        warnings=warnings,
    )
    assert selected is None
    assert "no conditioned_input" in warnings[0]


def test_local_source_discovery_keeps_every_supported_video(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator.evaluate import _resolve_source_clips

    inputs = tmp_path / "input"
    inputs.mkdir()
    expected = [inputs / "a.mp4", inputs / "b.mov", inputs / "c.webm"]
    for path in expected:
        path.write_bytes(b"video")
    (inputs / "ignore.txt").write_text("not video")
    selected = _resolve_source_clips(
        original_video="",
        input_uri=str(inputs),
        store=object(),
        workdir=tmp_path / "work",
        warnings=[],
    )
    assert selected == expected


def test_transient_attribute_endpoint_failure_marks_run_degraded(
    tmp_path: Path,
) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    _write_variant(augment, "clip-a", {"cloth_color": "blue"}, conditioned=False)
    result = evaluate_run(
        augment_uri=str(augment),
        output_uri=str(tmp_path / "grade"),
        client=_RateLimited([]),
        storage=object(),
    )
    assert result.status == "degraded"
    assert result.passed is False


@pytest.mark.parametrize(
    ("metadata_payload", "warning"),
    [([], "metadata is not an object"), ({"variables": []}, "variables are not an object")],
)
def test_malformed_variant_metadata_does_not_crash_and_degrades_run(
    tmp_path: Path, metadata_payload: Any, warning: str
) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    variant = tmp_path / "cosmos_augmented" / "clip-a"
    variant.mkdir(parents=True)
    (variant / "frame-000.png").write_bytes(b"frame")
    (variant / "metadata.json").write_text(json.dumps(metadata_payload))
    result = evaluate_run(
        augment_uri=str(tmp_path / "cosmos_augmented"),
        output_uri=str(tmp_path / "grade"),
        storage=object(),
    )
    assert result.status == "degraded"
    assert result.passed is False
    assert warning in result.warnings[0]


def test_attempt_scoped_evaluator_follows_only_the_committed_manifest(
    tmp_path: Path,
) -> None:
    import npa.workbench.cosmos_evaluator.evaluate as evaluator

    root = tmp_path / "cosmos_augmented"
    current = root / "_attempts" / "current" / "aug-current"
    old = root / "_attempts" / "old" / "aug-old"
    current.mkdir(parents=True)
    old.mkdir(parents=True)
    current_video = current / "augmented_video.mp4"
    current_video.write_bytes(b"current")
    (old / "augmented_video.mp4").write_bytes(b"old")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.cosmos2.transfer.v1",
                "mode": "cosmos_transfer2.5_gpu",
                "status": "executed",
                "node_count": 2,
                "attempt_id": "current",
                "scheduler_fence_sequence": 2,
                "scheduler_fence_attempt": 1,
                "scheduler_launch_id": "job",
                "publication_generation": 2,
                "logical_publication": "conditional",
                "logical_wave_id": "grade-loop-2",
                "membership_digest": "current-members",
                "variant_count": 1,
                "variants": [
                    {
                        "clip": "aug-current",
                        "variant_index": 0,
                        "augmented_video_uri": str(current_video),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert evaluator._list_clip_targets(str(root), store=object()) == [
        ("aug-current", str(current) + "/")
    ]


def test_evaluator_refuses_attempt_layout_without_canonical_manifest(
    tmp_path: Path,
) -> None:
    import npa.workbench.cosmos_evaluator.evaluate as evaluator

    root = tmp_path / "cosmos_augmented"
    (root / "_attempts" / "orphan" / "clip").mkdir(parents=True)
    with pytest.raises(evaluator.CosmosEvaluatorError, match="without a valid canonical"):
        evaluator._list_clip_targets(str(root), store=object())


def test_evaluator_follows_committed_cosmos3_manifest(tmp_path: Path) -> None:
    import npa.workbench.cosmos_evaluator.evaluate as evaluator
    from npa.workflows.paidf_cosmos3 import ENGINE, MANIFEST_SCHEMA

    root = tmp_path / "cosmos_augmented"
    videos = []
    variants = []
    for index, seed in enumerate((17, 18)):
        clip = f"variant-{index:04d}"
        video = root / clip / "augmented_video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(f"video-{index}".encode())
        videos.append(video)
        variants.append(
            {
                "clip": clip,
                "augmented_video_uri": str(video),
                "seed": seed,
                "video_bytes": video.stat().st_size,
                "frame_count": 8,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "engine": ENGINE,
                "status": "executed",
                "mode": "video2video",
                "input_conditioned": True,
                "input_conditioning": "source-video",
                "conditioned_input": "source.mp4",
                "guardrails": True,
                "weights_baked": False,
                "model": "Cosmos3-Nano",
                "lineage": {"input_provenance_uri": "input/provenance.json"},
                "variant_count": 2,
                "video_bytes": sum(item["video_bytes"] for item in variants),
                "frame_count": 16,
                "variants": variants,
            }
        ),
        encoding="utf-8",
    )

    assert evaluator._list_clip_targets(str(root), store=object()) == [
        (f"variant-{index:04d}", str(video.parent) + "/")
        for index, video in enumerate(videos)
    ]


def test_evaluator_refuses_cosmos3_manifest_with_reused_seed(tmp_path: Path) -> None:
    import npa.workbench.cosmos_evaluator.evaluate as evaluator
    from npa.workflows.paidf_cosmos3 import ENGINE, MANIFEST_SCHEMA

    root = tmp_path / "cosmos_augmented"
    variants = []
    for index in range(2):
        clip = f"variant-{index:04d}"
        video = root / clip / "augmented_video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        variants.append(
            {
                "clip": clip,
                "augmented_video_uri": str(video),
                "seed": 17,
                "video_bytes": 5,
                "frame_count": 8,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "engine": ENGINE,
                "status": "executed",
                "mode": "video2video",
                "input_conditioned": True,
                "input_conditioning": "source-video",
                "conditioned_input": "source.mp4",
                "guardrails": True,
                "weights_baked": False,
                "model": "Cosmos3-Nano",
                "lineage": {"input_provenance_uri": "input/provenance.json"},
                "variant_count": 2,
                "video_bytes": 10,
                "frame_count": 16,
                "variants": variants,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(evaluator.CosmosEvaluatorError, match="duplicated"):
        evaluator._list_clip_targets(str(root), store=object())


def write_report_helper(result: Any, tmp_path: Path) -> str:
    from npa.workbench.cosmos_evaluator.evaluate import write_report

    return write_report(result.to_dict(), result_uri=str(tmp_path / "grade"))
