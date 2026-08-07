"""Unit tests for the NVIDIA Cosmos Evaluator workbench tool.

Covers the algorithm helpers with no network or ffmpeg, the upstream-protocol
question/answer plumbing against a fake Token Factory client, and the run-level
report assembly against a fake storage client.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from npa.workbench.cosmos_evaluator import attribute_verification as av
from npa.workbench.cosmos_evaluator import hallucination as hal
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

    hallucinated, total = hal._hallucination_counts(original, augmented, dist_tol_px=7.0)
    assert total == 25  # 16 shared + 9 invented
    assert hallucinated == 9


def test_hallucination_counts_ignores_a_static_augmented_clip() -> None:
    original = np.zeros((8, 8), dtype=np.uint8)
    original[2:4, 2:4] = 255
    assert hal._hallucination_counts(original, np.zeros((8, 8), dtype=np.uint8), 7.0) == (0, 0)


def test_dynamic_mask_thresholds_frame_difference() -> None:
    previous = np.zeros((24, 24), dtype=np.uint8)
    current = np.zeros((24, 24), dtype=np.uint8)
    current[8:16, 8:16] = 200

    mask = hal._dynamic_mask(previous, current, grad_thresh=10.0, blur_ksize=1, morph_k=1)
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


def test_numpy_mask_path_matches_opencv_when_opencv_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NumPy kernels stand in for OpenCV's, so they must agree with them."""

    pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
    current = previous.copy()
    current[12:30, 12:30] = 240

    with_opencv = hal._dynamic_mask(previous, current, grad_thresh=10.0, blur_ksize=7, morph_k=3)
    monkeypatch.setattr(hal, "_cv2", lambda: None)
    with_numpy = hal._dynamic_mask(previous, current, grad_thresh=10.0, blur_ksize=7, morph_k=3)

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
        hal.check_hallucination(clip_id="c", original_video=clip, augmented_video=clip, threshold=1.5)


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
    parsed = av.parse_question_response(f"Sure, here you go:\n```json\n{payload}\n```\n")
    assert parsed["variable"] == "cloth_color"


def test_parse_question_response_drops_a_reasoning_block() -> None:
    payload = _question_json("cloth_color", "blue", "red")
    parsed = av.parse_question_response(f"<think>weighing options {{</think>{payload}")
    assert parsed["options"]["A"] == "blue"


def test_parse_question_response_unwraps_a_single_item_list() -> None:
    payload = json.loads(_question_json("lighting", "warm lamp light", "bright daylight"))
    assert av.parse_question_response(json.dumps([payload]))["variable"] == "lighting"


def test_parse_question_response_rejects_unparseable_output() -> None:
    with pytest.raises(CosmosEvaluatorError, match="could not parse"):
        av.parse_question_response("I would rather not answer.")


def test_normalize_question_repairs_a_wrong_answer_key() -> None:
    raw = json.loads(_question_json("cloth_color", "blue", "red", correct="B"))
    # The generator labelled B as correct while B is the *other* value.
    normalized = av.normalize_question(
        raw, variable="cloth_color", value="blue", options=APPEARANCE_OPTIONS["cloth_color"]
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
        av.normalize_question(raw, variable="cloth_color", value="blue", options=["red", "green"])


def test_normalize_question_rejects_a_single_option() -> None:
    raw = {
        "variable": "cloth_color",
        "value": "blue",
        "question": "What colour?",
        "options": {"A": "blue"},
        "correct_answer": "A",
    }
    with pytest.raises(CosmosEvaluatorError, match="options"):
        av.normalize_question(raw, variable="cloth_color", value="blue", options=["blue"])


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("A", "A"), ("The answer is C.", "C"), ("b", "B"), ("no idea", "UNKNOWN"), ("", "UNKNOWN")],
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


def test_verify_attributes_records_a_failing_check_without_dropping_the_batch(tmp_path: Path) -> None:
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
        client=client, variable="cloth_color", value="blue", options=["blue", "red"], model="m"
    )
    assert question["correct_answer"]


def test_a_rate_limit_is_not_retried_as_if_it_were_a_schema_problem() -> None:
    """Retrying a 429 doubles the load and hides the cause behind a JSON message."""

    client = _RateLimited([])
    with pytest.raises(RuntimeError, match="rate limit"):
        av.generate_question(
            client=client, variable="cloth_color", value="blue", options=["blue"], model="m"
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
        av.verify_attributes(clip_id="c", frame=tmp_path / "f.png", selected_variables={})


def test_format_question_lists_options_in_letter_order() -> None:
    text = av.format_question("What colour?", {"B": "red", "A": "blue"})
    assert text.splitlines()[1:3] == ["A) blue", "B) red"]


# ---------------------------------------------------------------------------
# Run-level report assembly
# ---------------------------------------------------------------------------


def _write_variant(root: Path, clip: str, variables: dict[str, str], *, conditioned: bool) -> None:
    variant = root / clip
    variant.mkdir(parents=True)
    (variant / "frame-000.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (variant / "metadata.json").write_text(
        json.dumps({"variables": variables, "input_conditioned": conditioned}), encoding="utf-8"
    )


def test_evaluate_run_grades_every_local_variant(tmp_path: Path) -> None:
    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    _write_variant(augment, "clip-a", {"cloth_color": "blue", "prompt": "ignored"}, conditioned=False)
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
        evaluate_run(augment_uri=str(empty), output_uri=str(tmp_path / "out"), storage=object())


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
    assert report_uri_for("s3://b/run/grade/custom.json") == "s3://b/run/grade/custom.json"


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
        contents = [
            {"Key": f"cosmos_augmented/{clip}/metadata.json"} for clip in clips
        ]
        self.s3 = SimpleNamespace(
            list_objects_v2=lambda **kwargs: {"Contents": contents, "IsTruncated": False}
        )

    def download_path(self, uri: str, dest: str) -> str:
        from botocore.exceptions import EndpointConnectionError

        raise EndpointConnectionError(endpoint_url="https://storage.invalid")


def test_a_storage_outage_is_reported_degraded_not_as_a_batch_of_zeros(tmp_path: Path) -> None:
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
    assert any("could not read" in warning for warning in result.warnings), result.warnings


def test_an_absent_variant_is_a_skip_not_an_outage(tmp_path: Path) -> None:
    """The mirror case: nothing to read is a legitimate zero, and stays completed."""

    from npa.workbench.cosmos_evaluator import evaluate_run

    augment = tmp_path / "cosmos_augmented"
    (augment / "clip-a").mkdir(parents=True)  # a variant dir with no artifacts in it

    result = evaluate_run(augment_uri=str(augment), output_uri=str(tmp_path / "grade"))
    assert result.status == "completed"
    assert result.score == 0.0
    assert result.clips[0].skipped


def write_report_helper(result: Any, tmp_path: Path) -> str:
    from npa.workbench.cosmos_evaluator.evaluate import write_report

    return write_report(result.to_dict(), result_uri=str(tmp_path / "grade"))
