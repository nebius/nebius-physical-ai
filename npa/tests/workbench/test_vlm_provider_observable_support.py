"""Pin provider-observable support validation, retention, and no-call boundaries."""

from __future__ import annotations

from dataclasses import asdict, fields
import json

import pytest

from npa.workbench import vlm_eval


_MISSING = object()
_VALID_SUPPORT = (
    " Frame 1 visibly shows a red square on the left. ",
    "Frame 2 visibly shows the same red square on the right — café.",
    "Frame 2 visibly shows the same red square on the right — café.",
)


def _payload(support: object = _VALID_SUPPORT) -> dict[str, object]:
    payload: dict[str, object] = {
        "success": True,
        "score": 0.9,
        "rationale": "banana",
    }
    if support is not _MISSING:
        payload["provider_observable_support"] = support
    return payload


def _parse(payload: dict[str, object], backend: str):
    text = json.dumps(payload, ensure_ascii=False)
    if backend == "hosted":
        return vlm_eval._parse_api_structured_response(text, served_model="model")
    return vlm_eval.parse_structured_response(text)


@pytest.mark.parametrize("backend", ["hosted", "self-hosted"])
def test_valid_support_is_preserved_after_json_decoding(
    backend: str,
) -> None:
    parsed = _parse(_payload(list(_VALID_SUPPORT)), backend)
    assert parsed.rationale == "banana"
    assert parsed.provider_observable_support == _VALID_SUPPORT
    expected_parser = (
        "npa_vlm_eval_hosted_json_v2"
        if backend == "hosted"
        else "npa_vlm_eval_compatible_json_v3"
    )
    assert parsed.parser_version == expected_parser


@pytest.mark.parametrize("backend", ["hosted", "self-hosted"])
@pytest.mark.parametrize(
    "support",
    [
        _MISSING,
        None,
        "Frame 1",
        False,
        1,
        {},
        [],
        [[]],
        [None],
        [False],
        [1],
        [{}],
        [""],
        [" \t\n"],
    ],
    ids=[
        "missing",
        "null",
        "string-scalar",
        "boolean",
        "number",
        "object",
        "empty-array",
        "nested-array",
        "null-member",
        "boolean-member",
        "number-member",
        "object-member",
        "empty-member",
        "blank-member",
    ],
)
def test_real_parsers_reject_missing_or_malformed_support(
    backend: str, support: object
) -> None:
    with pytest.raises(
        vlm_eval.VlmEvalError,
        match="provider_observable_support must be a nonempty array of nonempty strings",
    ):
        _parse(_payload(support), backend)


def _structured(
    support: tuple[str, ...] = _VALID_SUPPORT,
) -> vlm_eval.VlmStructuredResponse:
    return vlm_eval.VlmStructuredResponse(
        success=True,
        score=0.9,
        rationale="provider rationale",
        provider_success=True,
        provider_observable_support=support,
    )


def _result_from_structured(
    structured: vlm_eval.VlmStructuredResponse,
    *,
    provider_call_made: bool,
) -> vlm_eval.VlmEvalResult:
    return vlm_eval._result_from_structured(
        backend="api",
        input_path="rollout",
        output_path="output",
        task="task",
        model="model",
        success_threshold=0.8,
        frame_selection="sequence",
        frame_count=2,
        rubric="rubric",
        structured=structured,
        provider_call_made=provider_call_made,
    )


def test_direct_result_retains_support_without_semantic_validation() -> None:
    result = _result_from_structured(_structured(), provider_call_made=True)
    assert result.provider_observable_support == _VALID_SUPPORT
    assert result.passed is True
    assert vlm_eval._PROVIDER_OBSERVABLE_SUPPORT_LIMITATION in result.limitations


def test_real_result_rejects_empty_support() -> None:
    with pytest.raises(
        vlm_eval.VlmEvalError,
        match="Real-provider VLM results require provider_observable_support",
    ):
        _result_from_structured(_structured(()), provider_call_made=True)


def test_no_call_result_rejects_fabricated_support() -> None:
    with pytest.raises(
        vlm_eval.VlmEvalError,
        match="No-call VLM results cannot contain provider_observable_support",
    ):
        _result_from_structured(_structured(), provider_call_made=False)


def test_result_boundary_requires_immutable_support_tuple() -> None:
    structured = vlm_eval.VlmStructuredResponse(
        True,
        0.9,
        "rationale",
        provider_observable_support=["mutable"],  # type: ignore[arg-type]
    )
    with pytest.raises(
        vlm_eval.VlmEvalError,
        match="must be a tuple of nonempty strings",
    ):
        _result_from_structured(structured, provider_call_made=True)


def test_stub_and_override_do_not_fabricate_support() -> None:
    stub = vlm_eval.evaluate_stub(input_path="in", output_path="out", score=0.9)
    override = vlm_eval.evaluate_vlm(
        input_path="in", output_path="out", backend="api", score=0.9
    )
    assert stub.provider_observable_support == ()
    assert override.provider_observable_support == ()
    assert stub.provider_call_made is False
    assert override.provider_call_made is False


def test_new_fields_are_appended_for_positional_compatibility() -> None:
    assert fields(vlm_eval.VlmStructuredResponse)[-1].name == (
        "provider_observable_support"
    )
    assert fields(vlm_eval.VlmEvalResult)[-1].name == "provider_observable_support"
    assert fields(vlm_eval.VlmLoopRollout)[-1].name == "provider_observable_support"
    assert fields(vlm_eval.VlmBenchmarkCaseResult)[-1].name == (
        "provider_observable_support"
    )

    structured = vlm_eval.VlmStructuredResponse(
        True, 0.9, "rationale", None, None, "parser", True, False
    )
    result = vlm_eval.VlmEvalResult(
        "passed", "stub", "in", "out", "uri", "task", "model", 0.9, 0.8, True, "now"
    )
    rollout = vlm_eval.VlmLoopRollout(
        "id", True, 0.9, "rationale", "passed", 2, "uri", "model", None, False
    )
    case = vlm_eval.VlmBenchmarkCaseResult(
        "id",
        "rollout",
        True,
        True,
        0.9,
        "passed",
        True,
        "task",
        "rationale",
        2,
        "api",
        None,
        True,
        True,
        "model",
        "model",
        True,
    )
    assert structured.provider_observable_support == ()
    assert result.provider_observable_support == ()
    assert rollout.provider_observable_support == ()
    assert case.provider_observable_support == ()


def _provider_result() -> vlm_eval.VlmEvalResult:
    return vlm_eval.VlmEvalResult(
        status="passed",
        backend="api",
        input_path="rollout",
        output_path="out",
        result_uri="out/vlm_eval.json",
        task="task",
        model="model",
        score=0.9,
        success_threshold=0.8,
        passed=True,
        generated_at="now",
        frame_count=2,
        rationale="rationale",
        provider_call_made=True,
        provider_observable_support=_VALID_SUPPORT,
    )


def test_loop_projection_retains_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_eval, "discover_rollouts", lambda _path: ["rollout"])
    monkeypatch.setattr(vlm_eval, "evaluate_vlm", lambda **_kwargs: _provider_result())
    monkeypatch.setattr(
        vlm_eval,
        "write_result",
        lambda _payload, *, result_uri, storage_client=None: result_uri,
    )
    report = vlm_eval.evaluate_rollout_set(
        input_path="rollouts", output_path="results", backend="api"
    )
    assert report["rollouts"][0]["provider_observable_support"] == list(_VALID_SUPPORT)
    assert vlm_eval._PROVIDER_OBSERVABLE_SUPPORT_LIMITATION in report["limitations"]


def test_benchmark_projection_retains_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vlm_eval, "evaluate_vlm", lambda **_kwargs: _provider_result())
    case = vlm_eval._run_benchmark_case(
        vlm_eval.VlmBenchmarkItem("id", "rollout", True, "task"),
        config=vlm_eval.VlmBenchmarkConfig(
            "api", "model", "rubric", "rubric", 0.8, "sequence", 2
        ),
        endpoint_url="",
        api_key_env="KEY",
        timeout_s=1,
        use_fixture_score=False,
        selected_frames=(),
        preselected_task="task",
    )
    assert case.provider_observable_support == _VALID_SUPPORT


def test_paired_projection_and_json_persistence_retain_support(
    tmp_path,
) -> None:
    context = vlm_eval._VlmJudgeContext(
        input_path="rollout",
        output_path=str(tmp_path),
        task="task",
        rubric="rubric",
        success_threshold=0.8,
        frame_selection="sequence",
        max_frames=2,
        endpoint_url="",
        api_key_env="KEY",
        timeout_s=1,
        prompt="prompt",
        frames=(),
    )
    outcome = vlm_eval._comparison_success_outcome(
        model="model",
        request_sha256="a" * 64,
        structured=_structured(),
        context=context,
    )
    assert outcome.result is not None
    assert outcome.result.provider_observable_support == _VALID_SUPPORT
    assert (
        vlm_eval._PROVIDER_OBSERVABLE_SUPPORT_LIMITATION
        in vlm_eval._comparison_limitations()
    )
    assert (
        vlm_eval._PROVIDER_OBSERVABLE_SUPPORT_LIMITATION
        in vlm_eval._BENCHMARK_REPORT_LIMITATIONS
    )

    destination = tmp_path / "result.json"
    vlm_eval.write_result(asdict(outcome.result), result_uri=str(destination))
    payload = json.loads(destination.read_text())
    assert payload["provider_observable_support"] == list(_VALID_SUPPORT)
