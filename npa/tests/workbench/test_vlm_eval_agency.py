"""Paired outcome-versus-agency calibration and benchmark preflight tests."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image, ImageDraw
import pytest

from npa.workbench import vlm_eval
from npa.workbench.vlm_eval import (
    DEFAULT_ISAAC_AGENCY_BENCHMARK_PATH,
    DEFAULT_SAMPLE_BENCHMARK_PATH,
    SelectedFrame,
    VlmEvalError,
    VlmEvalResult,
    VlmStructuredResponse,
    benchmark_vlm_eval,
)
from npa.workbench.vlm_eval.agency import (
    AgencyStructuralError,
    evaluate_agency_structure,
    parse_agency_structural_check,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_FRAMES = REPO_ROOT / "docs/assets/hackathon/isaac-franka-lift-cube"
PACKAGED_FRAMES = DEFAULT_ISAAC_AGENCY_BENCHMARK_PATH.parent / (
    "rollouts/isaac-franka-lift-cube"
)
SOURCE_SHA256 = (
    "1f5f98000efd68f0ef909f869cbd945273cdc84b3a5b6aa59f1b9ea658cda1d0",
    "78ea93db5d592147a797ba39349bd66173bd77fc9d60810c9f387b9f5e8d882c",
    "fa4a17764d3c01778fed30376613dd6cb6bc0d4d07fee7fd55712b80f397e2b7",
    "83056e0acf266bfa7821383631f7231e7afd55dd84dd2c8e450e673e1c18aa33",
    "971f4b87dbee7c99b5b45545d701832f5e4678c5127325308b478ed96a5196e7",
    "18bf59b0feddb9b98802533e9188f0171c770d8415d8ffbe39642dceb1537341",
)
NORMALIZED_SHA256 = (
    "a11e411b701ecace52c4d324a5603435bf6577633be91fcc2f776e16e45a58af",
    "b687156a9d606c5410028c9f777387dd69cba0a25497bb03172c872f166fd6d8",
    "48b41db30e7b272617db230aa1594e4d9c2c16ef826651acfcfb132c25e97df9",
    "195af3fa11382ae0ce9f519d783cc3f36d7fdbfcf5ff49f4c0235a14793e0d20",
    "02fa3ac1b6dcf65f160e67d64073c0f913cf3b30ab55bd10cf0f696185d5bfcc",
    "bb1fa58e881d0083ffd35bf9461cc16dfdd9b119758f076db18597344de859a3",
)
DECODED_RGB_SHA256 = (
    "883a518a5efbb4369fbd2cd68a283073f3abe47b9ae5ec04078ed00f503ed4cb",
    "0dbbbe753251fbdbe63f096b41694f9906374649a04104255fbe8e5b5e890870",
    "d9be442e439e821f4732db595aa5414205acd0195f00e384c2ce2d47a34304ef",
    "6195dee34c8bdfb2a62664ef45a56a06d6ecb9edccb5679c10a7ef9f728fd3df",
    "b1adc37d160a96408a51537ee5f5de936ae1a93f1c46af664dddae7370fbd3ce",
    "c4f577ba371b9906fdcf5fc21ba3518692396699d572431316d1036f870ef6f5",
)


def test_packaged_frames_are_exact_source_bytes() -> None:
    source_paths = sorted(SOURCE_FRAMES.glob("frame_*.png"))
    packaged_paths = sorted(PACKAGED_FRAMES.glob("frame_*.png"))
    manifest = json.loads(
        (DEFAULT_ISAAC_AGENCY_BENCHMARK_PATH.parent / "frame-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(source_paths) == len(packaged_paths) == 6
    assert tuple(_sha256(path.read_bytes()) for path in source_paths) == SOURCE_SHA256
    assert tuple(path.read_bytes() for path in packaged_paths) == tuple(
        path.read_bytes() for path in source_paths
    )
    assert manifest["dimensions"] == [320, 240]
    assert tuple(frame["name"] for frame in manifest["frames"]) == tuple(
        path.name for path in packaged_paths
    )
    assert tuple(frame["bytes"] for frame in manifest["frames"]) == tuple(
        path.stat().st_size for path in packaged_paths
    )
    assert (
        tuple(frame["source_sha256"] for frame in manifest["frames"]) == SOURCE_SHA256
    )
    assert (
        tuple(frame["normalized_png_sha256"] for frame in manifest["frames"])
        == NORMALIZED_SHA256
    )
    assert (
        tuple(frame["decoded_rgb_sha256"] for frame in manifest["frames"])
        == DECODED_RGB_SHA256
    )


def test_isaac_agency_alias_measures_the_frozen_pair() -> None:
    report = benchmark_vlm_eval(
        dataset="isaac-agency",
        backend="stub",
        models=["fixture-stub"],
        rubrics=["default"],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )

    checks = report.sweep["structural_checks"]
    positive = checks["cube-elevated-positive"]
    negative = checks["robot-grasp-lift-negative"]
    assert positive["verdict"] == "pass"
    assert negative["verdict"] == "fail"
    assert positive["selected_labels"] == tuple(
        f"frame_{index:02d}.png" for index in range(6)
    )
    assert positive["normalized_png_sha256"] == NORMALIZED_SHA256
    assert positive["decoded_rgb_sha256"] == DECODED_RGB_SHA256
    assert positive["object_bottoms"] == (195, 195, 190, 175, 155, 135)
    assert positive["signed_first_to_last_vertical_rise_px"] == 60
    assert negative["horizontal_gaps"] == (50, 66, 82, 86, 90, 94)
    assert negative["minimum_gap_px"] == 50
    assert negative["object_motion_range_px"] == 60
    assert negative["dual_mask_coverage"] == 1.0


@pytest.mark.parametrize(
    ("frame_selection", "max_frames", "message"),
    [
        ("keyframes", 6, "requires frame_selection=sequence"),
        ("sequence", 5, "requires max_frames=6"),
    ],
)
def test_isaac_agency_alias_requires_the_exact_selection_contract(
    monkeypatch: pytest.MonkeyPatch,
    frame_selection: str,
    max_frames: int,
    message: str,
) -> None:
    monkeypatch.setattr(
        vlm_eval,
        "_normalize_backend",
        lambda *_args, **_kwargs: pytest.fail("backend normalization was called"),
    )

    with pytest.raises(VlmEvalError, match=message):
        benchmark_vlm_eval(
            dataset="isaac-agency",
            backend="api",
            frame_selection=frame_selection,
            max_frames=max_frames,
        )


def test_structural_preflight_reuses_exact_selected_frame_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[SelectedFrame, ...]] = []

    def fake_evaluate(**kwargs: Any) -> VlmEvalResult:
        selected = kwargs["_selected_frames"]
        received.append(selected)
        positive = kwargs["output_path"].endswith("cube-elevated-positive")
        return VlmEvalResult(
            status="passed" if positive else "needs_iteration",
            backend="self-hosted",
            input_path=kwargs["input_path"],
            output_path=kwargs["output_path"],
            result_uri=kwargs["output_path"],
            task=kwargs["task"],
            model=kwargs["model"],
            score=0.9 if positive else 0.1,
            success_threshold=kwargs["success_threshold"],
            passed=positive,
            generated_at="2026-10-01T00:00:00Z",
            frame_selection=kwargs["frame_selection"],
            frame_count=len(selected),
            rationale="identity control",
        )

    monkeypatch.setattr(vlm_eval, "evaluate_vlm", fake_evaluate)
    report = benchmark_vlm_eval(
        dataset="isaac-agency",
        backend="self-hosted",
        models=["test-model"],
        rubrics=["default"],
        thresholds=[0.5, 0.8],
        frame_selection="sequence",
        max_frames=6,
    )

    assert len(received) == 4
    assert all(candidate is received[0] for candidate in received)
    assert all(
        left is right
        for candidate in received[1:]
        for left, right in zip(received[0], candidate)
    )
    report_hashes = report.sweep["structural_checks"]["cube-elevated-positive"][
        "normalized_png_sha256"
    ]
    assert tuple(_sha256(frame.data) for frame in received[0]) == report_hashes


def test_evaluator_submits_preselected_objects_without_rediscovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = _frames(bottoms=(30, 22), gaps=(5, 5))
    submitted: list[tuple[SelectedFrame, ...]] = []
    monkeypatch.setattr(
        vlm_eval,
        "select_rollout_frames",
        lambda *_args, **_kwargs: pytest.fail("frames were rediscovered"),
    )

    def fake_call(**kwargs: Any) -> VlmStructuredResponse:
        submitted.append(kwargs["frames"])
        return VlmStructuredResponse(success=True, score=0.9, rationale="exact frames")

    monkeypatch.setattr(vlm_eval, "_call_openai_compatible", fake_call)
    result = vlm_eval.evaluate_vlm(
        input_path="does-not-exist",
        output_path=str(tmp_path / "result.json"),
        backend="self-hosted",
        task="frozen task",
        _selected_frames=frames,
    )

    assert submitted == [frames]
    assert submitted[0] is frames
    assert result.frame_count == 2


def test_structural_preflight_freezes_metadata_task_and_explicit_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rollout_root = tmp_path / "rollouts"
    rollout = rollout_root / "case"
    shutil.copytree(PACKAGED_FRAMES, rollout)
    (rollout / "info.json").write_text(
        json.dumps({"task": "metadata-derived task"}),
        encoding="utf-8",
    )
    payload = _packaged_manifest()
    payload["rollout_base_path"] = str(rollout_root)
    for item in payload["items"]:
        item["rollout"] = "case"
        item.pop("task", None)
    payload["items"][0]["task"] = "explicit state task"
    manifest = _write_payload(tmp_path, payload)

    original_select = vlm_eval.select_rollout_frames
    selected: list[tuple[SelectedFrame, ...]] = []
    prompts: list[str] = []

    def counted_select(*args: Any, **kwargs: Any) -> list[SelectedFrame]:
        frames = original_select(*args, **kwargs)
        selected.append(tuple(frames))
        return frames

    def fake_call(**kwargs: Any) -> VlmStructuredResponse:
        prompts.append(kwargs["prompt"])
        is_positive = "explicit state task" in kwargs["prompt"]
        return VlmStructuredResponse(
            success=is_positive,
            score=0.9 if is_positive else 0.1,
            rationale="task preservation control",
        )

    monkeypatch.setattr(vlm_eval, "select_rollout_frames", counted_select)
    monkeypatch.setattr(vlm_eval, "_call_openai_compatible", fake_call)
    report = benchmark_vlm_eval(
        dataset=str(manifest),
        backend="self-hosted",
        models=["test-model"],
        rubrics=["metadata preservation rubric"],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )

    assert len(selected) == 1
    assert len(prompts) == 2
    assert any("Task/instruction: explicit state task" in prompt for prompt in prompts)
    assert any(
        "Task/instruction: metadata-derived task" in prompt for prompt in prompts
    )
    assert all("Rubric: metadata preservation rubric" in prompt for prompt in prompts)
    assert {result.task for result in report.best_config.results} == {
        "explicit state task",
        "metadata-derived task",
    }

    stub_report = benchmark_vlm_eval(
        dataset=str(manifest),
        backend="stub",
        models=["fixture-stub"],
        rubrics=["metadata preservation rubric"],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )
    assert len(selected) == 2
    assert [result.task for result in stub_report.best_config.results] == [
        "explicit state task",
        "sim-to-real",
    ]


def test_structural_pair_propagates_custom_rubric_to_request_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rubrics: list[str] = []

    def fake_call(**kwargs: Any) -> VlmStructuredResponse:
        rubrics.append(kwargs["rubric"])
        assert f"Rubric: {kwargs['rubric']}" in kwargs["prompt"]
        assert kwargs["frame_selection"] == "sequence"
        assert kwargs["max_frames"] == 6
        return VlmStructuredResponse(success=False, score=0.1, rationale="control")

    monkeypatch.setattr(vlm_eval, "_call_openai_compatible", fake_call)
    benchmark_vlm_eval(
        dataset="isaac-agency",
        backend="api",
        models=["test-model"],
        rubrics=["Frozen custom agency rubric."],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )
    assert rubrics == ["Frozen custom agency rubric."] * 2


def test_structural_pair_retains_landed_sampling_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = {
        "model": "MiniMaxAI/MiniMax-M3",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "success": False,
                            "score": 0.25,
                            "rationale": "Synthetic metadata control.",
                        }
                    )
                },
            }
        ],
    }
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **kwargs: completion
    )
    report = benchmark_vlm_eval(
        dataset="isaac-agency",
        backend="api",
        models=["MiniMaxAI/MiniMax-M3"],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )
    assert len(report.best_config.results) == 2
    for case in report.best_config.results:
        assert case.evidence is not None
        assert case.evidence.schema_version == "npa_vlm_eval_evidence_v2"
        sampling = case.evidence.request.request_manifest["sampling"]
        assert sampling["strategy"] == "sequence"
        assert sampling["max_frames"] == sampling["selected_count"] == 6
        assert sampling["source_kind"] == "image-sequence"
        assert sampling["source_count"] == 6
        assert sampling["selected_indices"] == list(range(6))
        assert sampling["selected_timestamps_s"] == [None] * 6
        assert sampling["coverage_complete"] is True
        assert [frame.source_index for frame in case.evidence.request.frames] == list(
            range(6)
        )


@pytest.mark.parametrize(
    ("rise", "expected"),
    [(7, "fail"), (8, "pass")],
)
def test_object_elevation_uses_signed_inclusive_threshold(
    rise: int, expected: str
) -> None:
    frames = _frames(bottoms=(30, 30 - rise), gaps=(5, 5))
    result = evaluate_agency_structure(
        frames,
        _check(frames, claim="object_elevates"),
        frame_selection="sequence",
        max_frames=2,
    )
    assert result.verdict == expected
    assert result.signed_first_to_last_vertical_rise_px == rise


@pytest.mark.parametrize(
    "bottoms",
    [
        (22, 30),
        (30, 22, 30),
    ],
)
def test_object_elevation_rejects_descent_and_return(
    bottoms: tuple[int, ...],
) -> None:
    frames = _frames(bottoms=bottoms, gaps=(5,) * len(bottoms))
    result = evaluate_agency_structure(
        frames,
        _check(frames, claim="object_elevates"),
        frame_selection="sequence",
        max_frames=len(frames),
    )
    assert result.verdict == "fail"


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(0, "inconclusive"), (4, "inconclusive"), (5, "fail")],
)
def test_agency_proximity_never_establishes_grasp(gap: int, expected: str) -> None:
    frames = _frames(bottoms=(30, 22), gaps=(gap, gap))
    result = evaluate_agency_structure(
        frames,
        _check(frames, claim="actor_causes_motion"),
        frame_selection="sequence",
        max_frames=2,
    )
    assert result.verdict == expected


def test_agency_requires_complete_dual_masks() -> None:
    frames = _frames(
        bottoms=(30, 28, 26, 24, 22, 20),
        gaps=(5, 5, 5, 5, 5, 5),
        missing_actor={3},
    )
    result = evaluate_agency_structure(
        frames,
        _check(frames, claim="actor_causes_motion"),
        frame_selection="sequence",
        max_frames=6,
    )
    assert result.verdict == "inconclusive"
    assert result.dual_mask_coverage == pytest.approx(5 / 6)


def test_object_elevation_does_not_require_actor_masks() -> None:
    frames = _frames(
        bottoms=(30, 26, 22),
        gaps=(5, 5, 5),
        missing_actor={0, 1, 2},
    )
    result = evaluate_agency_structure(
        frames,
        _check(frames, claim="object_elevates"),
        frame_selection="sequence",
        max_frames=3,
    )
    assert result.verdict == "pass"
    assert result.actor_mask_count == 0
    assert result.object_endpoint_completeness == 1.0


def test_object_elevation_requires_both_ordered_endpoint_masks() -> None:
    frames = _frames(
        bottoms=(30, 26, 22),
        gaps=(5, 5, 5),
        missing_object={2},
    )
    result = evaluate_agency_structure(
        frames,
        _check(frames, claim="object_elevates"),
        frame_selection="sequence",
        max_frames=3,
    )
    assert result.verdict == "inconclusive"
    assert result.object_endpoint_completeness == 0.5


def test_agency_rejects_attribution_when_object_does_not_move() -> None:
    frames = _frames(bottoms=(30, 30), gaps=(5, 5))
    result = evaluate_agency_structure(
        frames,
        _check(frames, claim="actor_causes_motion"),
        frame_selection="sequence",
        max_frames=2,
    )
    assert result.verdict == "fail"
    assert result.object_motion_range_px == 0


def test_unknown_structural_kind_is_rejected() -> None:
    payload = _structural_payload()
    payload["kind"] = "unknown"

    with pytest.raises(AgencyStructuralError, match="unsupported.*kind"):
        parse_agency_structural_check(payload)


def test_unknown_structural_claim_is_rejected() -> None:
    payload = _structural_payload()
    payload["claim"] = "unknown"

    with pytest.raises(AgencyStructuralError, match="unsupported.*claim"):
        parse_agency_structural_check(payload)


def test_duplicate_required_labels_are_rejected() -> None:
    payload = _structural_payload()
    payload["required_labels"][1] = payload["required_labels"][0]

    with pytest.raises(AgencyStructuralError, match="labels must be unique"):
        parse_agency_structural_check(payload)


@pytest.mark.parametrize("digest", ["A" * 64, "g" * 64, "0" * 63])
def test_malformed_normalized_png_hashes_are_rejected(digest: str) -> None:
    payload = _structural_payload()
    payload["required_normalized_png_sha256"][0] = digest

    with pytest.raises(AgencyStructuralError, match="lowercase SHA-256"):
        parse_agency_structural_check(payload)


@pytest.mark.parametrize("mutation", ["unknown-field", "missing-field"])
def test_malformed_structural_check_is_rejected(mutation: str) -> None:
    payload = _structural_payload()
    if mutation == "unknown-field":
        payload["unreviewed_threshold"] = 1
        expected = "unknown fields"
    else:
        del payload["movement_px_inclusive"]
        expected = "missing fields"

    with pytest.raises(AgencyStructuralError, match=expected):
        parse_agency_structural_check(payload)


@pytest.mark.parametrize(
    "field",
    [
        "required_max_frames",
        "title_rows_excluded",
        "object_r_min_inclusive",
        "object_g_max_inclusive",
        "object_b_max_inclusive",
        "actor_channel_min_inclusive",
        "actor_rg_delta_max_inclusive",
        "actor_gb_delta_max_inclusive",
        "contact_px_inclusive",
        "movement_px_inclusive",
    ],
)
@pytest.mark.parametrize("value", [False, True])
def test_structural_integer_fields_reject_booleans(field: str, value: bool) -> None:
    payload = _structural_payload()
    payload[field] = value

    with pytest.raises(AgencyStructuralError, match=f"{field} must be an integer"):
        parse_agency_structural_check(payload)


@pytest.mark.parametrize("frame_selection", ["final", "keyframes", "bogus"])
def test_structural_parser_requires_sequence(frame_selection: str) -> None:
    payload = _structural_payload()
    payload["required_frame_selection"] = frame_selection

    with pytest.raises(
        AgencyStructuralError,
        match="required_frame_selection must be sequence",
    ):
        parse_agency_structural_check(payload)


@pytest.mark.parametrize(
    ("first_id", "second_id"),
    [
        ("item-001", "item-001"),
        (None, "item-001"),
    ],
)
def test_duplicate_resolved_ids_fail_before_frame_or_evaluator_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_id: str | None,
    second_id: str,
) -> None:
    items = [
        _manifest_item(first_id, "pass"),
        _manifest_item(second_id, "fail"),
    ]
    manifest = _write_manifest(tmp_path, items)
    monkeypatch.setattr(
        vlm_eval,
        "select_rollout_frames",
        lambda *_args, **_kwargs: pytest.fail("frame selection was called"),
    )
    monkeypatch.setattr(
        vlm_eval,
        "evaluate_vlm",
        lambda **_kwargs: pytest.fail("evaluator was called"),
    )
    monkeypatch.setattr(
        vlm_eval,
        "_resolve_api_key",
        lambda **_kwargs: pytest.fail("credential resolution was called"),
    )
    monkeypatch.setattr(
        vlm_eval,
        "_call_openai_compatible",
        lambda **_kwargs: pytest.fail("provider was called"),
    )

    with pytest.raises(
        VlmEvalError,
        match="duplicate benchmark item id 'item-001'.*must be unique",
    ):
        benchmark_vlm_eval(dataset=str(manifest), backend="api")


def test_structured_duplicate_ids_fail_before_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _packaged_manifest()
    payload["items"][1]["id"] = payload["items"][0]["id"]
    manifest = _write_payload(tmp_path, payload)
    monkeypatch.setattr(
        vlm_eval,
        "select_rollout_frames",
        lambda *_args, **_kwargs: pytest.fail("frame selection was called"),
    )
    monkeypatch.setattr(
        vlm_eval,
        "evaluate_agency_structure",
        lambda *_args, **_kwargs: pytest.fail("structural evaluation was called"),
    )

    with pytest.raises(VlmEvalError, match="duplicate benchmark item id"):
        benchmark_vlm_eval(
            dataset=str(manifest),
            backend="api",
            frame_selection="sequence",
            max_frames=6,
        )


@pytest.mark.parametrize("label", ["pass", "fail"])
def test_single_class_dataset_fails_before_frame_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, label: str
) -> None:
    manifest = _write_manifest(
        tmp_path,
        [_manifest_item("item-a", label), _manifest_item("item-b", label)],
    )
    monkeypatch.setattr(
        vlm_eval,
        "select_rollout_frames",
        lambda *_args, **_kwargs: pytest.fail("frame selection was called"),
    )
    monkeypatch.setattr(
        vlm_eval,
        "evaluate_vlm",
        lambda **_kwargs: pytest.fail("evaluator was called"),
    )

    with pytest.raises(VlmEvalError, match="at least one pass and one fail"):
        benchmark_vlm_eval(dataset=str(manifest), backend="api")


def test_positive_actor_claim_fails_during_parse_before_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _packaged_manifest()
    payload["items"][1]["expected_label"] = "pass"
    payload["items"].append(_manifest_item("balance-negative", "fail"))
    manifest = _write_payload(tmp_path, payload)
    for name in (
        "select_rollout_frames",
        "evaluate_agency_structure",
        "_normalize_backend",
        "_resolve_api_key",
        "_call_openai_compatible",
    ):
        monkeypatch.setattr(
            vlm_eval,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"{_name} was called"),
        )

    with pytest.raises(
        VlmEvalError,
        match=(
            "benchmark item 2.*actor_causes_motion.*"
            "refutation-only.*expected_label fail"
        ),
    ):
        benchmark_vlm_eval(
            dataset=str(manifest),
            backend="api",
            frame_selection="sequence",
            max_frames=6,
        )


def test_object_label_mismatch_still_fails_structural_preflight(
    tmp_path: Path,
) -> None:
    payload = _packaged_manifest()
    payload["items"][0]["expected_label"] = "fail"
    payload["items"].append(_manifest_item("balance-positive", "pass"))
    manifest = _write_payload(tmp_path, payload)

    with pytest.raises(VlmEvalError, match="does not match expected_label"):
        benchmark_vlm_eval(
            dataset=str(manifest),
            backend="stub",
            frame_selection="sequence",
            max_frames=6,
        )


def test_invalid_later_structural_item_aborts_before_backend_or_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _packaged_manifest()
    payload["items"][1]["structural_check"]["required_normalized_png_sha256"][-1] = (
        "0" * 64
    )
    manifest = _write_payload(tmp_path, payload)
    monkeypatch.setattr(
        vlm_eval,
        "_normalize_backend",
        lambda *_args, **_kwargs: pytest.fail("backend normalization was called"),
    )
    monkeypatch.setattr(
        vlm_eval,
        "evaluate_vlm",
        lambda **_kwargs: pytest.fail("evaluator was called"),
    )

    with pytest.raises(VlmEvalError, match="normalized PNG hashes"):
        benchmark_vlm_eval(
            dataset=str(manifest),
            backend="api",
            frame_selection="sequence",
            max_frames=6,
        )


def test_legacy_benchmark_has_no_structural_report_key() -> None:
    report = benchmark_vlm_eval(
        dataset=str(DEFAULT_SAMPLE_BENCHMARK_PATH),
        backend="stub",
        models=["fixture-stub"],
        rubrics=["default"],
        thresholds=[0.8],
    )

    assert "structural_checks" not in report.sweep
    assert report.best_config.metrics.specificity == 1.0
    assert report.best_config.metrics.balanced_accuracy == 1.0


def test_packaged_balanced_fixture_exposes_stub_all_yes_specificity() -> None:
    report = benchmark_vlm_eval(
        dataset="sample",
        backend="stub",
        models=["fixture-stub"],
        rubrics=["default"],
        thresholds=[0.0],
    )

    metrics = report.best_config.metrics
    assert metrics.true_positives == 2
    assert metrics.false_positives == 2
    assert metrics.true_negatives == 0
    assert metrics.false_negatives == 0
    assert metrics.recall == 1.0
    assert metrics.specificity == 0.0
    assert metrics.balanced_accuracy == 0.5


def test_packaged_all_positive_subset_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_SAMPLE_BENCHMARK_PATH.read_text(encoding="utf-8"))
    payload["rollout_base_path"] = str(
        DEFAULT_SAMPLE_BENCHMARK_PATH.parent / "rollouts"
    )
    payload["items"] = [
        item for item in payload["items"] if item["id"].endswith("pass")
    ]

    with pytest.raises(VlmEvalError, match="at least one pass and one fail"):
        benchmark_vlm_eval(
            dataset=str(_write_payload(tmp_path, payload)), backend="stub"
        )


def test_balanced_accuracy_ranks_above_raw_accuracy(tmp_path: Path) -> None:
    items = []
    for index in range(28):
        score = 0.9 if index < 14 else 0.7
        items.append(_manifest_item(f"positive-{index}", "pass", score=score))
    for index in range(4):
        items.append(_manifest_item(f"negative-{index}", "fail", score=0.6))
    manifest = _write_manifest(tmp_path, items)

    report = benchmark_vlm_eval(
        dataset=str(manifest),
        backend="stub",
        models=["fixture-stub"],
        rubrics=["default"],
        thresholds=[0.5, 0.8],
    )

    assert report.best_config.config.success_threshold == 0.8
    assert report.best_config.metrics.accuracy == 0.5625
    assert report.best_config.metrics.specificity == 1.0
    assert report.best_config.metrics.balanced_accuracy == 0.75
    all_yes = report.ranked_configs[1].metrics
    assert all_yes.accuracy == 0.875
    assert all_yes.specificity == 0.0
    assert all_yes.balanced_accuracy == 0.5


def test_specificity_and_balanced_accuracy_use_confusion_denominators(
    tmp_path: Path,
) -> None:
    items = [
        _manifest_item("positive-pass", "pass", score=0.9),
        _manifest_item("positive-miss", "pass", score=0.1),
        _manifest_item("negative-false-positive", "fail", score=0.9),
        _manifest_item("negative-a", "fail", score=0.1),
        _manifest_item("negative-b", "fail", score=0.1),
        _manifest_item("negative-c", "fail", score=0.1),
    ]
    report = benchmark_vlm_eval(
        dataset=str(_write_manifest(tmp_path, items)),
        backend="stub",
        models=["fixture-stub"],
        rubrics=["default"],
        thresholds=[0.5],
    )

    metrics = report.best_config.metrics
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.specificity == 0.75
    assert metrics.balanced_accuracy == 0.625


def _structural_payload() -> dict[str, Any]:
    frames = _frames(bottoms=(30, 22), gaps=(5, 5))
    payload = asdict(_check(frames, claim="object_elevates"))
    payload["required_labels"] = list(payload["required_labels"])
    payload["required_normalized_png_sha256"] = list(
        payload["required_normalized_png_sha256"]
    )
    return payload


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frames(
    *,
    bottoms: tuple[int, ...],
    gaps: tuple[int, ...],
    missing_actor: set[int] | None = None,
    missing_object: set[int] | None = None,
) -> tuple[SelectedFrame, ...]:
    actor_missing = missing_actor or set()
    object_missing = missing_object or set()
    frames = []
    for index, (bottom, gap) in enumerate(zip(bottoms, gaps)):
        image = Image.new("RGB", (64, 64), (10, 30, 60))
        draw = ImageDraw.Draw(image)
        actor_right = 20
        if index not in actor_missing:
            draw.rectangle((12, 12, actor_right, 28), fill=(180, 180, 180))
        if index not in object_missing:
            object_left = actor_right + gap
            draw.rectangle(
                (object_left, bottom - 5, object_left + 5, bottom),
                fill=(200, 20, 20),
            )
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        frames.append(
            SelectedFrame(
                label=f"frame_{index:02d}.png",
                media_type="image/png",
                data=buffer.getvalue(),
            )
        )
    return tuple(frames)


def _check(frames: tuple[SelectedFrame, ...], *, claim: str):
    return parse_agency_structural_check(
        {
            "kind": "stylized_color_bbox_agency_v1",
            "claim": claim,
            "required_frame_selection": "sequence",
            "required_max_frames": len(frames),
            "required_labels": [frame.label for frame in frames],
            "required_normalized_png_sha256": [_sha256(frame.data) for frame in frames],
            "title_rows_excluded": 0,
            "object_r_min_inclusive": 121,
            "object_g_max_inclusive": 79,
            "object_b_max_inclusive": 79,
            "actor_channel_min_inclusive": 151,
            "actor_rg_delta_max_inclusive": 17,
            "actor_gb_delta_max_inclusive": 17,
            "contact_px_inclusive": 4,
            "movement_px_inclusive": 8,
        }
    )


def _manifest_item(
    item_id: str | None, label: str, *, score: float = 0.9
) -> dict[str, Any]:
    item = {
        "rollout": "unused",
        "task": "calibration task",
        "expected_label": label,
        "fixture_score": score,
    }
    if item_id is not None:
        item["id"] = item_id
    return item


def _write_manifest(root: Path, items: list[dict[str, Any]]) -> Path:
    manifest = root / "benchmark.json"
    manifest.write_text(
        json.dumps({"format": "npa_vlm_eval_benchmark_v1", "items": items}),
        encoding="utf-8",
    )
    return manifest


def _packaged_manifest() -> dict[str, Any]:
    payload = json.loads(
        DEFAULT_ISAAC_AGENCY_BENCHMARK_PATH.read_text(encoding="utf-8")
    )
    payload["rollout_base_path"] = str(PACKAGED_FRAMES.parent)
    return payload


def _write_payload(root: Path, payload: dict[str, Any]) -> Path:
    manifest = root / "benchmark.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest
