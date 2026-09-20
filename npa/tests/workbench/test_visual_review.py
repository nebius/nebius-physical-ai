from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import threading
import traceback
from typing import Callable

import pytest
from PIL import Image

from npa.clients.storage import StoragePreconditionFailed
from npa.workbench import vlm_eval
from npa.workbench.vlm_eval import visual_review


MODEL = "MiniMaxAI/MiniMax-M3"


def _assertion(frame_id: str, text: str = "Visible scene evidence.") -> dict:
    return {"text": text, "frame_ids": [frame_id]}


def _arm(
    frame_id: str,
    *,
    task_status: str = "complete",
    fidelity_status: str = "no_visible_issue",
    reviewability: str = "reviewable",
    impressiveness: str = "moderate",
    usefulness: str = "plausible",
) -> dict:
    return {
        "task_evidence": {
            "visible_status": task_status,
            "observations": [_assertion(frame_id)],
            "hidden_state_limits": ["Pixels do not establish hidden state."],
        },
        "artifact_fidelity": _fidelity(frame_id, fidelity_status),
        "reviewability": _reviewability(frame_id, reviewability),
        "impressiveness": {
            "status": impressiveness,
            "visible_basis": [_assertion(frame_id)],
            "cosmetic_only": False,
        },
        "physical_ai_usefulness": _usefulness(frame_id, usefulness),
    }


def _fidelity(frame_id: str, status: str) -> dict:
    issues = []
    if status == "issues_visible":
        issues = [
            {
                "category": "temporal_defect",
                "severity": "minor",
                "assertion": _assertion(frame_id),
            }
        ]
    return {
        "status": status,
        "issues": issues,
        "uncertainty": "Unseen views remain unknown.",
    }


def _reviewability(frame_id: str, status: str) -> dict:
    limitations = [] if status == "reviewable" else [_assertion(frame_id)]
    strengths = [] if status == "unreviewable" else [_assertion(frame_id)]
    return {"status": status, "strengths": strengths, "limitations": limitations}


def _usefulness(frame_id: str, status: str) -> dict:
    supported = status != "unsupported"
    return {
        "status": status,
        "visible_basis": [_assertion(frame_id)],
        "downstream_operation": "rollout inspection" if supported else None,
        "required_properties": ["representative views"] if supported else [],
        "hypothesis": "May aid bounded review." if supported else None,
        "measured_consumer_test_needed": (
            "Measure reviewer agreement." if supported else None
        ),
    }


def _single_payload(frame_id: str = "A0001") -> dict:
    return {"arm": _arm(frame_id)}


def _paired_payload(
    *,
    current_is_A: bool,
    second_current_status: str = "complete",
    preferred: str | None = None,
    confidence: str = "high",
) -> dict:
    current_id = "A0001" if current_is_A else "B0001"
    baseline_id = "B0001" if current_is_A else "A0001"
    preferred_set = preferred or ("A" if current_is_A else "B")
    A_arm = _arm(current_id, task_status=second_current_status)
    B_arm = _arm(baseline_id, task_status="partial")
    if not current_is_A:
        A_arm, B_arm = B_arm, A_arm
    return {
        "A": A_arm,
        "B": B_arm,
        "comparison": {
            "preferred_set": preferred_set,
            "visible_evidence_difference": "materially_better",
            "confidence": confidence,
            "observations": [
                {
                    "text": "The visible evidence differs between the sets.",
                    "A_frame_ids": ["A0001"],
                    "B_frame_ids": ["B0001"],
                }
            ],
        },
    }


def _completion(
    payload: object,
    *,
    model: object = MODEL,
    finish: object = "stop",
    refusal: object | None = None,
    usage: object | None = None,
) -> dict:
    message = {"content": payload if isinstance(payload, str) else json.dumps(payload)}
    if refusal is not None:
        message["refusal"] = refusal
    response = {
        "id": "private-provider-id",
        "model": model,
        "choices": [{"finish_reason": finish, "message": message}],
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _response(completion: dict, *, raw_body: str | None = None):
    return vlm_eval._VlmBackendResponse(
        data=completion,
        raw_body=raw_body or vlm_eval._canonical_json(completion),
        status_code=200,
        request_id_header="private-header-id",
        latency_s=0.2,
    )


def _install_transport(monkeypatch, completions, calls=None) -> list[dict]:
    recorded = calls if calls is not None else []
    responses = iter(completions)

    def post(*, request, response_sink, **_kwargs):
        recorded.append(request)
        response = _response(next(responses))
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(
        visual_review,
        "_resolve_provider_key",
        lambda *_args, **_kwargs: "auth-secret",
    )
    return recorded


def _write_frames(root: Path, count: int = 1, color: str = "green") -> Path:
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (12, 9), color).save(root / f"frame-{index:03d}.png")
    return root


def _request(
    tmp_path: Path, *, paired: bool = False
) -> vlm_eval.VlmVisualReviewRequest:
    current = _write_frames(tmp_path / "source", count=5)
    baseline = _write_frames(tmp_path / "reference", color="blue") if paired else None
    return vlm_eval.VlmVisualReviewRequest(
        input_path=str(current),
        baseline_path=str(baseline) if baseline else "",
        output_path=str(tmp_path / "private-review"),
        model=MODEL,
        task="Assess visible placement evidence.",
        max_frames=3,
    )


@pytest.mark.parametrize("status", visual_review._TASK_STATUSES)
def test_parser_accepts_every_task_status(status: str) -> None:
    payload = _single_payload()
    payload["arm"]["task_evidence"]["visible_status"] = status

    verdict = vlm_eval.parse_visual_review_response(
        json.dumps(payload), mode="single", A_frame_ids=["A0001"]
    )

    assert verdict.arm.task_evidence.visible_status == status


@pytest.mark.parametrize("status", visual_review._FIDELITY_STATUSES)
def test_parser_accepts_every_fidelity_status(status: str) -> None:
    payload = {"arm": _arm("A0001", fidelity_status=status)}

    verdict = vlm_eval.parse_visual_review_response(
        json.dumps(payload), mode="single", A_frame_ids=["A0001"]
    )

    assert verdict.arm.artifact_fidelity.status == status


@pytest.mark.parametrize("category", visual_review._FIDELITY_CATEGORIES)
def test_parser_accepts_every_fidelity_category(category: str) -> None:
    payload = {"arm": _arm("A0001", fidelity_status="issues_visible")}
    payload["arm"]["artifact_fidelity"]["issues"][0]["category"] = category

    verdict = _parse_single(payload)

    assert verdict.arm.artifact_fidelity.issues[0].category == category


@pytest.mark.parametrize("severity", visual_review._ISSUE_SEVERITIES)
def test_parser_accepts_every_fidelity_severity(severity: str) -> None:
    payload = {"arm": _arm("A0001", fidelity_status="issues_visible")}
    payload["arm"]["artifact_fidelity"]["issues"][0]["severity"] = severity

    verdict = _parse_single(payload)

    assert verdict.arm.artifact_fidelity.issues[0].severity == severity


@pytest.mark.parametrize("status", visual_review._REVIEWABILITY_STATUSES)
def test_parser_accepts_every_reviewability_status(status: str) -> None:
    payload = {"arm": _arm("A0001", reviewability=status)}

    verdict = vlm_eval.parse_visual_review_response(
        json.dumps(payload), mode="single", A_frame_ids=["A0001"]
    )

    assert verdict.arm.reviewability.status == status


@pytest.mark.parametrize("status", visual_review._IMPRESSIVENESS_STATUSES)
def test_parser_accepts_every_impressiveness_status(status: str) -> None:
    payload = {"arm": _arm("A0001", impressiveness=status)}

    verdict = vlm_eval.parse_visual_review_response(
        json.dumps(payload), mode="single", A_frame_ids=["A0001"]
    )

    assert verdict.arm.impressiveness.status == status


@pytest.mark.parametrize("status", visual_review._USEFULNESS_STATUSES)
def test_usefulness_is_code_bounded_for_every_status(status: str) -> None:
    payload = {"arm": _arm("A0001", usefulness=status)}

    verdict = vlm_eval.parse_visual_review_response(
        json.dumps(payload), mode="single", A_frame_ids=["A0001"]
    )

    usefulness = verdict.arm.physical_ai_usefulness
    assert usefulness.status == status
    assert usefulness.confirmation_status == "hypothesis_only"


@pytest.mark.parametrize(
    ("preferred", "difference"),
    [
        ("A", "materially_better"),
        ("B", "materially_better"),
        ("tie", "equivalent"),
        ("unresolved", "unresolved"),
    ],
)
def test_parser_accepts_comparison_truth_table(preferred, difference) -> None:
    payload = _paired_payload(current_is_A=True, preferred=preferred)
    payload["comparison"]["visible_evidence_difference"] = difference

    verdict = vlm_eval.parse_visual_review_response(
        json.dumps(payload),
        mode="paired",
        A_frame_ids=["A0001"],
        B_frame_ids=["B0001"],
    )

    assert verdict.comparison.preferred_set == preferred


@pytest.mark.parametrize("confidence", visual_review._CONFIDENCES)
def test_parser_accepts_every_comparison_confidence(confidence: str) -> None:
    payload = _paired_payload(current_is_A=True, confidence=confidence)

    verdict = vlm_eval.parse_visual_review_response(
        json.dumps(payload),
        mode="paired",
        A_frame_ids=["A0001"],
        B_frame_ids=["B0001"],
    )

    assert verdict.comparison.confidence == confidence


def _mutations() -> list[tuple[str, Callable[[dict], None]]]:
    return [
        ("top-extra", lambda value: value.update({"score": 0.9})),
        ("top-missing", lambda value: value.pop("arm")),
        ("arm-extra", lambda value: value["arm"].update({"success": True})),
        ("dimension-missing", lambda value: value["arm"].pop("reviewability")),
        (
            "nested-extra",
            lambda value: value["arm"]["task_evidence"].update({"extra": "x"}),
        ),
        (
            "assertion-extra",
            lambda value: value["arm"]["task_evidence"]["observations"][0].update(
                {"extra": "x"}
            ),
        ),
        (
            "empty-observations",
            lambda value: value["arm"]["task_evidence"].update({"observations": []}),
        ),
        (
            "wrong-type",
            lambda value: value["arm"]["impressiveness"].update({"cosmetic_only": 0}),
        ),
        (
            "provider-confirmation",
            lambda value: value["arm"]["physical_ai_usefulness"].update(
                {"confirmation_status": "hypothesis_only"}
            ),
        ),
    ]


@pytest.mark.parametrize(("_name", "mutate"), _mutations())
def test_parser_rejects_schema_mutations(_name, mutate) -> None:
    payload = _single_payload()
    mutate(payload)

    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.parse_visual_review_response(
            json.dumps(payload), mode="single", A_frame_ids=["A0001"]
        )


def _paired_structure_mutations() -> list[tuple[str, Callable[[dict], None]]]:
    return [
        ("top-extra", lambda value: value.update({"score": 1})),
        ("top-missing", lambda value: value.pop("B")),
        ("arm-extra", lambda value: value["A"].update({"objective": True})),
        ("arm-missing", lambda value: value["B"].pop("reviewability")),
        (
            "dimension-extra",
            lambda value: value["A"]["task_evidence"].update({"extra": "x"}),
        ),
        (
            "dimension-missing",
            lambda value: value["B"]["artifact_fidelity"].pop("uncertainty"),
        ),
        ("comparison-missing", lambda value: value["comparison"].pop("confidence")),
        (
            "comparison-extra",
            lambda value: value["comparison"].update({"extra": "x"}),
        ),
        (
            "comparison-empty-list",
            lambda value: value["comparison"].update({"observations": []}),
        ),
    ]


def _paired_assertion_mutations() -> list[tuple[str, Callable[[dict], None]]]:
    return [
        (
            "assertion-extra",
            lambda value: value["A"]["task_evidence"]["observations"][0].update(
                {"extra": "x"}
            ),
        ),
        (
            "assertion-empty",
            lambda value: value["B"]["task_evidence"]["observations"][0].update(
                {"text": ""}
            ),
        ),
        (
            "comparison-assertion-extra",
            lambda value: value["comparison"]["observations"][0].update(
                {"frame_ids": ["A0001"]}
            ),
        ),
        (
            "comparison-assertion-missing",
            lambda value: value["comparison"]["observations"][0].pop("B_frame_ids"),
        ),
        (
            "comparison-assertion-empty",
            lambda value: value["comparison"]["observations"][0].update(
                {"A_frame_ids": []}
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("_name", "mutate"),
    _paired_structure_mutations() + _paired_assertion_mutations(),
)
def test_parser_rejects_paired_nested_schema_mutations(_name, mutate) -> None:
    payload = deepcopy(_paired_payload(current_is_A=True))
    mutate(payload)

    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.parse_visual_review_response(
            json.dumps(payload),
            mode="paired",
            A_frame_ids=["A0001"],
            B_frame_ids=["B0001"],
        )


@pytest.mark.parametrize(
    "content",
    [
        "```json\n" + json.dumps(_single_payload()) + "\n```",
        "prefix " + json.dumps(_single_payload()),
        json.dumps(_single_payload()) + " suffix",
        '["not","an","object"]',
        '{"arm":{"task_evidence":{"visible_status":"complete",'
        '"visible_status":"failure"}}}',
        '{"arm":{"task_evidence":{},"task_evidence":{}}}',
    ],
)
def test_parser_rejects_noncanonical_or_duplicate_json(content: str) -> None:
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.parse_visual_review_response(
            content, mode="single", A_frame_ids=["A0001"]
        )


@pytest.mark.parametrize(
    "frame_ids",
    [["A9999"], ["B0001"], ["A0001", "A0001"], [], "A0001"],
)
def test_parser_rejects_invalid_citations(frame_ids) -> None:
    payload = _single_payload()
    payload["arm"]["task_evidence"]["observations"][0]["frame_ids"] = frame_ids

    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.parse_visual_review_response(
            json.dumps(payload), mode="single", A_frame_ids=["A0001"]
        )


def test_parser_rejects_cross_field_and_dimension_overlap() -> None:
    payload = _single_payload()
    payload["arm"]["artifact_fidelity"]["status"] = "issues_visible"
    with pytest.raises(vlm_eval.VlmVisualReviewError, match="requires a visible issue"):
        _parse_single(payload)

    payload = _single_payload()
    payload["arm"]["artifact_fidelity"]["issues"] = [
        {
            "category": "blur",
            "severity": "minor",
            "assertion": _assertion("A0001"),
        }
    ]
    with pytest.raises(vlm_eval.VlmVisualReviewError, match="enum"):
        _parse_single(payload)


def _parse_single(payload: dict):
    return vlm_eval.parse_visual_review_response(
        json.dumps(payload), mode="single", A_frame_ids=["A0001"]
    )


def test_parser_rejects_established_usefulness_claim() -> None:
    payload = _single_payload()
    payload["arm"]["physical_ai_usefulness"]["hypothesis"] = (
        "Safety is certified by these pixels."
    )

    with pytest.raises(vlm_eval.VlmVisualReviewError, match="established benefit"):
        _parse_single(payload)


def test_parser_rejects_comparison_without_both_arm_citations() -> None:
    payload = _paired_payload(current_is_A=True)
    payload["comparison"]["observations"][0]["B_frame_ids"] = ["A0001"]

    with pytest.raises(vlm_eval.VlmVisualReviewError, match="unsubmitted"):
        vlm_eval.parse_visual_review_response(
            json.dumps(payload),
            mode="paired",
            A_frame_ids=["A0001"],
            B_frame_ids=["B0001"],
        )


def test_single_review_is_neutral_non_gating_and_retains_sampling(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path)
    usage = {"prompt_tokens": 17, "completion_tokens": 23, "total_tokens": 40}
    completion = _completion(_single_payload(), usage=usage)
    calls = _install_transport(monkeypatch, [completion])

    report = vlm_eval.review_visual(request)

    assert len(calls) == 1
    _assert_neutral_request(calls[0], expected_ids=["A0001", "A0002", "A0003"])
    assert report.status == "completed"
    assert report.deployment_status == "audit_only"
    assert report.score_gate_affected is False
    assert report.normalized_task_completion_score is None
    assert asdict(report.baseline_comparison) == {
        "current_vs_baseline": "not_provided",
        "agreement_eligible": False,
        "observations": (),
    }
    assert report.current_manifest.selected_indices == (0, 2, 4)
    assert report.current_manifest.coverage_complete is True
    assert report.current_review is not None
    assert report.current_review.physical_ai_usefulness.confirmation_status == (
        "hypothesis_only"
    )
    _assert_positive_provenance(report, calls[0], completion, usage)
    _assert_private_artifacts(report, tmp_path / "private-review")


def _assert_neutral_request(request: dict, *, expected_ids: list[str]) -> None:
    content = request["messages"][0]["content"]
    markers = [part["text"] for part in content if part["type"] == "text"][1:]
    assert markers == expected_ids
    for marker_index in range(1, len(content), 2):
        assert content[marker_index]["type"] == "text"
        assert content[marker_index + 1]["type"] == "image_url"
    text = visual_review._request_text(request)
    assert visual_review._SOURCE_ROLE_PATTERN.search(text) is None
    assert "frame-00" not in text
    assert "max_tokens" not in request


def _assert_positive_provenance(report, request, completion, usage) -> None:
    outcome = report.outcomes[0]
    request_evidence = outcome.request
    provider = outcome.provider
    submitted = _request_arm_hashes(request)["A"]
    assert outcome.transport_request_sha256 == vlm_eval._sha256_json(request)
    assert request_evidence.request_manifest_sha256 == vlm_eval._sha256_json(
        request_evidence.request_manifest
    )
    assert [frame.sha256 for frame in request_evidence.frames] == submitted
    assert [frame.sha256 for frame in report.current_manifest.frames] == submitted
    assert all(
        frame.width > 0 and frame.height > 0 for frame in request_evidence.frames
    )
    assert provider is not None
    assert provider.provider_request_id == "private-provider-id"
    assert provider.returned_model == MODEL
    assert provider.finish_reason == "stop"
    assert provider.usage == usage
    raw = vlm_eval._canonical_json(completion)
    assert provider.raw_response == raw
    assert provider.raw_response_sha256 == vlm_eval._sha256_text(raw)


def _assert_private_artifacts(report, output: Path) -> None:
    canonical = output / vlm_eval.VISUAL_REVIEW_RESULT_FILENAME
    evidence = output / visual_review.VISUAL_REVIEW_EVIDENCE_DIRECTORY
    assert report.result_uri == str(canonical)
    assert canonical.stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    assert evidence.stat().st_mode & 0o777 == 0o700
    expected = {
        "reservation.json",
        "request-01.json",
        "transport-started-01.json",
        "response-01.json",
        "outcome-01.json",
    }
    assert expected <= {path.name for path in evidence.iterdir()}
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in evidence.iterdir())


def test_source_role_guard_uses_standalone_words(monkeypatch, tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path),
        task="Assess currently visible concurrent motion.",
    )
    calls = _install_transport(monkeypatch, [_completion(_single_payload())])

    report = vlm_eval.review_visual(request)

    prompt = visual_review._request_text(calls[0])
    assert report.status == "completed"
    assert "currently visible concurrent motion" in prompt
    assert visual_review._SOURCE_ROLE_PATTERN.search(prompt) is None


@pytest.mark.parametrize(
    "private_text",
    ["Bearer private-token", "access_token=private", "x-amz-signature=private"],
)
def test_provider_text_rejects_authorization_before_reservation(
    monkeypatch, tmp_path: Path, private_text: str
) -> None:
    request = replace(_request(tmp_path), task=f"Inspect pixels. {private_text}")
    monkeypatch.setattr(
        vlm_eval,
        "_post_comparison_request",
        lambda **_kwargs: pytest.fail("transport must not run"),
    )

    with pytest.raises(vlm_eval.VlmVisualReviewError) as failure:
        vlm_eval.review_visual(request)

    _assert_bounded_traceback(failure.value, private_text)
    assert not _evidence_path(request).exists()


def test_prompt_matches_single_and_paired_parser_contracts() -> None:
    single = visual_review._review_prompt(
        "Inspect visible pixels.", "Stay bounded.", "single"
    )
    paired = visual_review._review_prompt(
        "Inspect visible pixels.", "Stay bounded.", "paired"
    )

    assert "top-level key set is exactly {arm}" in single
    assert 'sole key "arm"' in single
    assert "direct arm fields at top level are invalid" in single
    assert "nonempty observations" in single
    assert "issues_visible requires one or more issues" in single
    assert "unreviewable requires no strengths" in single
    assert "Single mode accepts no comparison object" in single
    assert "top-level key set is exactly {A, B, comparison}" in paired
    assert "comparison.observations must contain one or more objects" in paired
    assert "nonempty unique A_frame_ids and B_frame_ids" in paired
    assert "confirmation field" in single


def test_paired_review_counterbalances_and_maps_visible_improvement(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path, paired=True)
    calls = _install_transport(
        monkeypatch,
        [
            _completion(_paired_payload(current_is_A=True)),
            _completion(_paired_payload(current_is_A=False)),
        ],
    )

    report = vlm_eval.review_visual(request)

    assert len(calls) == 2
    first_hashes = _request_arm_hashes(calls[0])
    second_hashes = _request_arm_hashes(calls[1])
    assert first_hashes["A"] == second_hashes["B"]
    assert first_hashes["B"] == second_hashes["A"]
    assert (
        calls[0]["messages"][0]["content"][0] == calls[1]["messages"][0]["content"][0]
    )
    assert report.status == "completed"
    assert report.escalation_required is False
    assert report.baseline_comparison.current_vs_baseline == "materially_better"
    assert report.baseline_comparison.agreement_eligible is True
    assert len(report.baseline_comparison.observations) == 2
    assert report.current_review.task_evidence.visible_status == "complete"
    assert report.baseline_review.task_evidence.visible_status == "partial"


def _request_arm_hashes(request: dict) -> dict[str, list[str]]:
    hashes = {"A": [], "B": []}
    arm = ""
    for part in request["messages"][0]["content"]:
        if part["type"] == "text" and part["text"][:1] in hashes:
            arm = part["text"][0]
        if part["type"] != "image_url" or not arm:
            continue
        encoded = part["image_url"]["url"].split(",", 1)[1]
        hashes[arm].append(hashlib.sha256(base64.b64decode(encoded)).hexdigest())
    return hashes


def test_paired_dimension_disagreement_requires_escalation(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path, paired=True)
    _install_transport(
        monkeypatch,
        [
            _completion(_paired_payload(current_is_A=True)),
            _completion(
                _paired_payload(current_is_A=False, second_current_status="unclear")
            ),
        ],
    )

    report = vlm_eval.review_visual(request)

    assert report.status == "dimension_disagreement"
    assert report.escalation_required is True
    assert report.current_review is None
    assert report.baseline_comparison.current_vs_baseline == "unresolved"
    assert report.baseline_comparison.agreement_eligible is False


def test_paired_contract_error_keeps_first_outcome_and_runs_second(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path, paired=True)
    invalid = _completion("```json\n{}\n```")
    calls = _install_transport(
        monkeypatch,
        [invalid, _completion(_paired_payload(current_is_A=False))],
    )

    report = vlm_eval.review_visual(request)

    assert len(calls) == 2
    assert report.status == "judge_error"
    assert report.outcomes[0].error is not None
    assert report.outcomes[0].provider is not None
    assert report.outcomes[1].verdict is not None
    evidence = _evidence_path(request)
    assert (evidence / "outcome-01.json").exists()
    assert (evidence / "outcome-02.json").exists()


@pytest.mark.parametrize(
    "completion",
    [
        _completion(_single_payload(), model="vendor/wrong"),
        _completion(_single_payload(), finish="length"),
        _completion(_single_payload(), refusal="cannot inspect"),
        _completion("```json\n{}\n```"),
        _completion("not-json"),
        _completion(_single_payload(), usage=["wrong"]),
    ],
)
def test_provider_contract_errors_are_retained_generically(
    monkeypatch, tmp_path: Path, completion: dict
) -> None:
    request = _request(tmp_path)
    _install_transport(monkeypatch, [completion])

    report = vlm_eval.review_visual(request)

    outcome = report.outcomes[0]
    assert report.status == "judge_error"
    assert outcome.error is not None
    assert outcome.error.message == (
        "hosted visual review response violated the strict contract"
    )
    assert outcome.provider is not None
    assert outcome.provider.raw_response
    assert "private-provider-id" not in outcome.error.message


def test_raw_response_is_journaled_before_parser_runs(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path)
    raw = '{"id":"private-before-parse","choices":[]}'

    def post(*, response_sink, **_kwargs):
        response = _response(_completion(_single_payload()), raw_body=raw)
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )
    monkeypatch.setattr(
        visual_review,
        "parse_visual_review_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parse crash")),
    )

    with pytest.raises(vlm_eval.VlmVisualReviewError) as failure:
        vlm_eval.review_visual(request)

    response = _evidence_path(request) / "response-01.json"
    assert json.loads(response.read_text())["raw_body"] == raw
    _assert_bounded_traceback(failure.value, "parse crash")


def _evidence_path(request: vlm_eval.VlmVisualReviewRequest) -> Path:
    return Path(request.output_path) / visual_review.VISUAL_REVIEW_EVIDENCE_DIRECTORY


def test_post_start_crash_permanently_blocks_replay(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path)
    calls = 0

    def crash(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("simulated transport crash")

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", crash)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )

    with pytest.raises(vlm_eval.VlmVisualReviewError) as failure:
        vlm_eval.review_visual(request)
    _assert_bounded_traceback(failure.value, "transport crash")
    assert (_evidence_path(request) / "transport-started-01.json").exists()
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)
    assert calls == 1


@pytest.mark.parametrize(
    ("boundary", "expected_name"),
    [
        ("reservation", "reservation.json"),
        ("pre_start", "reservation.json"),
        ("post_outcome", "outcome-01.json"),
    ],
)
def test_crash_boundaries_consume_output_identity(
    monkeypatch, tmp_path: Path, boundary: str, expected_name: str
) -> None:
    request = _request(tmp_path)
    _install_transport(monkeypatch, [_completion(_single_payload())])
    if boundary == "reservation":
        monkeypatch.setattr(
            visual_review,
            "_resolve_provider_key",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("reservation crash")
            ),
        )
    elif boundary == "pre_start":
        monkeypatch.setattr(
            visual_review,
            "_run_attempts",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("pre-start crash")),
        )
    else:
        monkeypatch.setattr(
            visual_review,
            "_finalize_visual_review_report",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("finalization crash")
            ),
        )

    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)
    assert (_evidence_path(request) / expected_name).exists()
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)


def test_two_local_callers_make_at_most_one_single_attempt(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path)
    lock = threading.Lock()
    calls = 0

    def post(*, response_sink, **_kwargs):
        nonlocal calls
        with lock:
            calls += 1
        response = _response(_completion(_single_payload()))
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(vlm_eval.review_visual, request) for _ in range(2)]
    results = [_future_result(future) for future in futures]

    assert calls <= 1
    assert sum(not isinstance(result, Exception) for result in results) == 1


def test_two_local_callers_make_at_most_two_paired_attempts(
    monkeypatch, tmp_path: Path
) -> None:
    request = _request(tmp_path, paired=True)
    completions = iter(
        [
            _completion(_paired_payload(current_is_A=True)),
            _completion(_paired_payload(current_is_A=False)),
        ]
    )
    lock = threading.Lock()
    calls = []

    def post(*, response_sink, **_kwargs):
        with lock:
            completion = next(completions)
            calls.append(completion)
        response = _response(completion)
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(vlm_eval.review_visual, request) for _ in range(2)]
    results = [_future_result(future) for future in futures]

    assert len(calls) <= 2
    assert sum(not isinstance(result, Exception) for result in results) == 1


def _future_result(future):
    try:
        return future.result()
    except Exception as exc:
        return exc


def _assert_bounded_traceback(error: Exception, sentinel: str) -> None:
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert str(error) == "Visual review failed; inspect private evidence."
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert sentinel not in rendered


class _ConditionalStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.lock = threading.Lock()

    def put_bytes_conditional(self, payload, uri, **kwargs):
        assert kwargs == {
            "if_none_match": True,
            "content_type": "application/json",
        }
        with self.lock:
            if uri in self.objects:
                raise StoragePreconditionFailed("lost race")
            self.objects[uri] = payload
        return '"etag"'

    def read_bytes_with_etag(self, uri):
        with self.lock:
            payload = self.objects.get(uri)
        return (payload, '"etag"') if payload is not None else None


def test_two_s3_callers_make_at_most_two_paired_attempts(
    monkeypatch, tmp_path: Path
) -> None:
    base = _request(tmp_path, paired=True)
    request = vlm_eval.VlmVisualReviewRequest(
        **{**asdict(base), "output_path": "s3://private-bucket/review/"}
    )
    storage = _ConditionalStorage()
    responses = [
        _completion(_paired_payload(current_is_A=True)),
        _completion(_paired_payload(current_is_A=False)),
    ]
    response_lock = threading.Lock()
    calls = 0

    def post(*, response_sink, **_kwargs):
        nonlocal calls
        with response_lock:
            completion = responses[calls]
            calls += 1
        response = _response(completion)
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(vlm_eval.review_visual, request, storage_client=storage)
            for _ in range(2)
        ]
    results = [_future_result(future) for future in futures]

    assert calls <= 2
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert (
        request.output_path + vlm_eval.VISUAL_REVIEW_RESULT_FILENAME in storage.objects
    )
    _assert_s3_evidence_objects(request.output_path, storage.objects)


def test_two_s3_callers_make_at_most_one_single_attempt(
    monkeypatch, tmp_path: Path
) -> None:
    base = _request(tmp_path)
    request = replace(base, output_path="s3://private-bucket/single-review/")
    storage = _ConditionalStorage()
    calls = []

    def post(*, response_sink, **_kwargs):
        calls.append(True)
        response = _response(_completion(_single_payload()))
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(vlm_eval.review_visual, request, storage_client=storage)
            for _ in range(2)
        ]
    results = [_future_result(future) for future in futures]

    assert len(calls) <= 1
    assert sum(not isinstance(result, Exception) for result in results) == 1


def test_preseeded_s3_report_stops_before_credentials_and_transport(
    monkeypatch, tmp_path: Path
) -> None:
    request = replace(_request(tmp_path), output_path="s3://private-bucket/preseed/")
    storage = _ConditionalStorage()
    canonical = request.output_path + vlm_eval.VISUAL_REVIEW_RESULT_FILENAME
    storage.objects[canonical] = b'{"preseeded":true}'
    credentials_called = False
    transport_called = False

    def credentials(*_args, **_kwargs):
        nonlocal credentials_called
        credentials_called = True

    def post(**_kwargs):
        nonlocal transport_called
        transport_called = True

    monkeypatch.setattr(visual_review, "_resolve_provider_key", credentials)
    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request, storage_client=storage)

    root = request.output_path + visual_review.VISUAL_REVIEW_EVIDENCE_DIRECTORY
    assert f"{root}/reservation.json" in storage.objects
    assert credentials_called is False
    assert transport_called is False
    assert storage.objects[canonical] == b'{"preseeded":true}'


def test_s3_consumed_transport_start_is_never_replayed(
    monkeypatch, tmp_path: Path
) -> None:
    request = replace(_request(tmp_path), output_path="s3://private-bucket/crash/")
    storage = _ConditionalStorage()
    calls = 0

    def crash(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("private-s3-crash")

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", crash)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request, storage_client=storage)
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request, storage_client=storage)

    root = request.output_path + visual_review.VISUAL_REVIEW_EVIDENCE_DIRECTORY
    assert f"{root}/transport-started-01.json" in storage.objects
    assert calls == 1


def _assert_s3_evidence_objects(output_path: str, objects: dict[str, bytes]) -> None:
    root = output_path + visual_review.VISUAL_REVIEW_EVIDENCE_DIRECTORY
    expected = {
        f"{root}/reservation.json",
        f"{root}/transport-started-01.json",
        f"{root}/response-01.json",
        f"{root}/outcome-01.json",
        f"{root}/transport-started-02.json",
        f"{root}/response-02.json",
        f"{root}/outcome-02.json",
    }
    assert expected <= set(objects)


def _symlink_cases(root: Path) -> list[Path]:
    cases = []
    root.mkdir()
    real = root / "real"
    real.mkdir(mode=0o700)
    parent_link = root / "parent-link"
    parent_link.symlink_to(real, target_is_directory=True)
    cases.append(parent_link / vlm_eval.VISUAL_REVIEW_RESULT_FILENAME)
    for name in ("destination", "temporary", "evidence"):
        parent = root / name
        parent.mkdir(mode=0o700)
        target = parent / "target"
        target.write_text("private")
        if name == "destination":
            link = parent / vlm_eval.VISUAL_REVIEW_RESULT_FILENAME
        elif name == "temporary":
            link = parent / f".{vlm_eval.VISUAL_REVIEW_RESULT_FILENAME}.tmp"
        else:
            link = parent / visual_review.VISUAL_REVIEW_EVIDENCE_DIRECTORY
        link.symlink_to(target, target_is_directory=name == "evidence")
        cases.append(parent / vlm_eval.VISUAL_REVIEW_RESULT_FILENAME)
    return cases


def test_symlinked_output_components_are_rejected_before_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    source = _write_frames(tmp_path / "source")
    credentials_called = False

    def credentials(*_args, **_kwargs):
        nonlocal credentials_called
        credentials_called = True
        return "secret"

    monkeypatch.setattr(visual_review, "_resolve_provider_key", credentials)
    for index, output in enumerate(_symlink_cases(tmp_path / "links")):
        request = vlm_eval.VlmVisualReviewRequest(
            input_path=str(source),
            output_path=str(output),
            model=MODEL,
            task=f"Inspect visible evidence {index}.",
        )
        with pytest.raises(vlm_eval.VlmVisualReviewError):
            vlm_eval.review_visual(request)
    assert credentials_called is False


def test_alternate_json_filename_fails_before_sampling(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        vlm_eval,
        "select_rollout_frames",
        lambda *_args, **_kwargs: pytest.fail("sampling must not run"),
    )
    request = vlm_eval.VlmVisualReviewRequest(
        input_path=str(tmp_path / "missing"),
        output_path=str(tmp_path / "other.json"),
        model=MODEL,
        task="Inspect visible evidence.",
    )

    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)


def test_private_metadata_never_enters_transport_or_auth_journals(
    monkeypatch, tmp_path: Path
) -> None:
    objective = tmp_path / "objective.json"
    objective.write_text('{"references":["s3://private/objective.json"]}')
    matched = tmp_path / "matched.json"
    matched.write_text('{"pairs":[{"left":0,"right":0}]}')
    base = _request(tmp_path)
    request = vlm_eval.VlmVisualReviewRequest(
        **{
            **asdict(base),
            "objective_evidence_path": str(objective),
            "matched_view_map_path": str(matched),
            "endpoint_url": "https://private-endpoint.invalid/v1",
        }
    )
    calls = _install_transport(monkeypatch, [_completion(_single_payload())])

    report = vlm_eval.review_visual(request)

    transport = json.dumps(calls)
    assert "objective.json" not in transport
    assert "matched.json" not in transport
    evidence_text = "\n".join(
        path.read_text() for path in _evidence_path(request).iterdir()
    )
    assert "auth-secret" not in evidence_text
    assert "private-endpoint.invalid" not in evidence_text
    assert report.objective_evidence_status == "references_provided_unverified"
    assert report.matched_view_status == "provided_unverified"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@example.invalid/v1",
        "https://example.invalid/v1?token=secret",
        "https://example.invalid/v1#private",
    ],
)
def test_unsafe_endpoint_fails_before_reservation(endpoint, tmp_path: Path) -> None:
    request = _request(tmp_path)
    request = vlm_eval.VlmVisualReviewRequest(
        **{**asdict(request), "endpoint_url": endpoint}
    )

    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)

    assert not _evidence_path(request).exists()


def test_default_endpoint_ignores_openai_key_for_token_factory(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import token_factory

    request = replace(_request(tmp_path), api_key_env="OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-ambient-key")
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    resolved = []
    monkeypatch.setattr(
        token_factory,
        "resolve_config",
        lambda **kwargs: resolved.append(kwargs) or _KeyConfig("file-key"),
    )
    headers = {}

    def post(*, response_sink, **kwargs):
        headers.update(kwargs["headers"])
        response = _response(_completion(_single_payload()))
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    assert vlm_eval.review_visual(request).status == "completed"
    assert headers["Authorization"] == "Bearer file-key"
    assert "wrong-ambient-key" not in json.dumps(headers)
    assert resolved == [
        {
            "api_key_env": token_factory.DEFAULT_API_KEY_ENV,
            "require_api_key": False,
        }
    ]


def test_default_endpoint_honors_exact_non_openai_environment(
    monkeypatch, tmp_path: Path
) -> None:
    request = replace(_request(tmp_path), api_key_env="PRIVATE_TF_KEY")
    monkeypatch.setenv("PRIVATE_TF_KEY", "exact-token-factory-key")
    headers = {}

    def post(*, response_sink, **kwargs):
        headers.update(kwargs["headers"])
        response = _response(_completion(_single_payload()))
        response_sink(response)
        return response, None

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)

    assert vlm_eval.review_visual(request).status == "completed"
    assert headers["Authorization"] == "Bearer exact-token-factory-key"


class _KeyConfig:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


def test_explicit_endpoint_requires_exact_environment_before_transport(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import token_factory

    request = replace(
        _request(tmp_path),
        endpoint_url="https://private-credential-sentinel.invalid/v1",
        api_key_env="EXACT_VISUAL_REVIEW_KEY",
    )
    monkeypatch.delenv("EXACT_VISUAL_REVIEW_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "wrong-token-factory-key")
    monkeypatch.setattr(
        token_factory,
        "resolve_config",
        lambda **_kwargs: pytest.fail("explicit endpoint cannot use file fallback"),
    )
    transport_called = False

    def post(**_kwargs):
        nonlocal transport_called
        transport_called = True

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    with pytest.raises(vlm_eval.VlmVisualReviewError) as failure:
        vlm_eval.review_visual(request)

    _assert_bounded_traceback(failure.value, "private-credential-sentinel")
    assert transport_called is False
    assert (_evidence_path(request) / "reservation.json").exists()
    assert not (_evidence_path(request) / "transport-started-01.json").exists()


@pytest.mark.parametrize(
    "alias",
    [
        "access_token",
        "refresh-token",
        "idToken",
        "AWS_ACCESS_KEY_ID",
        "aws-secret-access-key",
        "aws_session_token",
        "securityToken",
        "private_secret",
        "client-secret",
        "signed_url",
        "presigned-url",
        "x-amz-signature",
    ],
)
def test_sensitive_metadata_aliases_are_rejected_recursively(
    monkeypatch, tmp_path: Path, alias: str
) -> None:
    matched = tmp_path / "matched.json"
    matched.write_text(json.dumps({"outer": [{"metadata": {alias: "private"}}]}))
    request = replace(_request(tmp_path), matched_view_map_path=str(matched))
    transport_called = False

    def post(**_kwargs):
        nonlocal transport_called
        transport_called = True

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(
        visual_review, "_resolve_provider_key", lambda *_args, **_kwargs: "secret"
    )
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)

    assert transport_called is False
    assert not _evidence_path(request).exists()


def test_sensitive_presigned_url_value_is_rejected_recursively(
    monkeypatch, tmp_path: Path
) -> None:
    matched = tmp_path / "matched.json"
    matched.write_text(
        json.dumps(
            {"outer": [{"locator": "https://example.invalid/x?X-Amz-Signature=secret"}]}
        )
    )
    request = replace(_request(tmp_path), matched_view_map_path=str(matched))
    monkeypatch.setattr(
        vlm_eval,
        "_post_comparison_request",
        lambda **_kwargs: pytest.fail("transport must not run"),
    )

    with pytest.raises(vlm_eval.VlmVisualReviewError):
        vlm_eval.review_visual(request)


def test_report_finalizer_is_private_and_rejects_forged_reports(
    monkeypatch, tmp_path: Path
) -> None:
    assert "write_visual_review_report" not in vlm_eval.__all__
    assert not hasattr(vlm_eval, "write_visual_review_report")
    request = _request(tmp_path)
    _install_transport(monkeypatch, [_completion(_single_payload())])
    captured = {}
    real_finalize = visual_review._finalize_visual_review_report

    def capture(report, context, journal, outcomes):
        captured.update(
            report=report, context=context, journal=journal, outcomes=outcomes
        )

    monkeypatch.setattr(visual_review, "_finalize_visual_review_report", capture)
    report = vlm_eval.review_visual(request)
    args = (captured["context"], captured["journal"], captured["outcomes"])
    with pytest.raises(vlm_eval.VlmVisualReviewError):
        real_finalize(asdict(report), *args)  # type: ignore[arg-type]
    current = report.current_review
    assert current is not None
    usefulness = replace(current.physical_ai_usefulness, confirmation_status="measured")
    forged_review = replace(current, physical_ai_usefulness=usefulness)
    forged = [
        replace(report, score_gate_affected=True),
        replace(report, normalized_task_completion_score=0.9),
        replace(report, objective_evidence_status="verified"),
        replace(report, result_uri="s3://forged/report.json"),
        replace(report, current_review=forged_review),
    ]
    for value in forged:
        with pytest.raises(vlm_eval.VlmVisualReviewError):
            real_finalize(value, *args)
    real_finalize(report, *args)
    assert Path(report.result_uri).is_file()
