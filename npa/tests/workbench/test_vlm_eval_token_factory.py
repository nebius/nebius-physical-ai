from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from npa.workbench.vlm_eval import (
    VlmEvalError,
    _resolve_api_key,
    _resolve_endpoint_url,
    benchmark_vlm_eval,
    evaluate_vlm,
)


def test_api_backend_defaults_to_token_factory_served_vision_model(tmp_path) -> None:
    """The Token Factory API serves MiniMax-M3, not vlm_eval's self-hosted
    default (Qwen2-VL-7B, which 404s), so the api backend must pick the served
    model unless --model is overridden."""
    from npa.clients.token_factory import DEFAULT_VISION_MODEL

    result = evaluate_vlm(
        input_path="s3://ignored",
        output_path=str(tmp_path / "out.json"),
        backend="api",
        score=0.9,  # skips the real VLM call
    )
    assert result.model == DEFAULT_VISION_MODEL


def test_api_backend_defaults_to_token_factory_base_url(monkeypatch) -> None:
    for key in (
        "VLM_EVAL_API_BASE_URL",
        "OPENAI_BASE_URL",
        "NEBIUS_TOKEN_FACTORY_BASE_URL",
        "NEBIUS_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    url = _resolve_endpoint_url(backend="api", endpoint_url="")
    assert url == "https://api.tokenfactory.nebius.com/v1/"


def test_api_backend_honors_explicit_endpoint(monkeypatch) -> None:
    url = _resolve_endpoint_url(backend="api", endpoint_url="http://localhost:9000/v1")
    assert url == "http://localhost:9000/v1"


def test_api_backend_accepts_token_factory_key(monkeypatch) -> None:
    monkeypatch.delenv("VLM_EVAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "tf-key")
    assert _resolve_api_key(backend="api", api_key_env="VLM_EVAL_API_KEY") == "tf-key"


def test_api_backend_requires_a_key(monkeypatch) -> None:
    from npa.clients import token_factory

    for key in ("VLM_EVAL_API_KEY", "NEBIUS_TOKEN_FACTORY_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        token_factory,
        "resolve_config",
        lambda **kwargs: SimpleNamespace(api_key=""),
    )
    with pytest.raises(VlmEvalError):
        _resolve_api_key(backend="api", api_key_env="VLM_EVAL_API_KEY")


@pytest.mark.parametrize(
    ("model", "constrained"),
    [
        ("MiniMaxAI/MiniMax-M3", False),
        ("vendor/explicit-vision", True),
    ],
)
def test_api_judge_uses_model_specific_json_mode(
    monkeypatch, model, constrained
) -> None:
    from npa.workbench import vlm_eval

    requests = []

    def post(**kwargs):
        requests.append(kwargs["request"])
        return _completion(model=model)

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    result = vlm_eval._call_openai_compatible(
        backend="api",
        model=model,
        endpoint_url="https://example.test/v1",
        api_key_env="TEST_KEY",
        prompt="Return JSON",
        frames=[],
        timeout_s=120,
    )
    assert result.score == 0.9
    assert ("response_format" in requests[0]) is constrained
    if not constrained:
        assert requests[0]["chat_template_kwargs"] == {"thinking_mode": "disabled"}


def test_malformed_minimax_json_remains_an_error(monkeypatch) -> None:
    from npa.workbench import vlm_eval

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval,
        "_post_with_readiness_retry",
        lambda **kwargs: _completion(content='{"{"}success":true,"score":1}'),
    )
    with pytest.raises(VlmEvalError, match="JSON could not be parsed"):
        vlm_eval._call_openai_compatible(
            backend="api",
            model="MiniMaxAI/MiniMax-M3",
            endpoint_url="https://example.test/v1",
            api_key_env="TEST_KEY",
            prompt="Return JSON",
            frames=[],
            timeout_s=120,
        )


_VALID_CONTENT = '{"success":true,"score":0.9,"rationale":"target reached"}'


def _completion(*, content=_VALID_CONTENT, model="MiniMaxAI/MiniMax-M3", finish="stop"):
    return {
        "model": model,
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
    }


def _call_completion(
    monkeypatch, completion, *, backend="api", model="MiniMaxAI/MiniMax-M3"
):
    from npa.workbench import vlm_eval

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **kwargs: completion
    )
    return vlm_eval._call_openai_compatible(
        backend=backend,
        model=model,
        endpoint_url="https://example.test/v1",
        api_key_env="TEST_KEY",
        prompt="Return JSON",
        frames=[],
        timeout_s=120,
    )


@pytest.mark.parametrize(
    "score", [7, -0.1, float("nan"), float("inf"), True, "0.9", None]
)
def test_api_judge_rejects_invalid_scores_before_result(monkeypatch, score) -> None:
    content = json.dumps(
        {"success": True, "score": score, "rationale": "target reached"}
    )
    with pytest.raises(VlmEvalError, match="finite number|non-finite number"):
        _call_completion(monkeypatch, _completion(content=content))


@pytest.mark.parametrize(
    "content",
    [
        '{"success":"yes","score":0.9,"rationale":"target reached"}',
        '{"score":0.9,"rationale":"target reached"}',
        '{"success":true,"score":0.1,"score":0.9,"rationale":"target reached"}',
        "Evaluation: " + _VALID_CONTENT,
        _VALID_CONTENT + " trailing explanation",
        '{"success":true,"score":0.9}',
        '{"success":true,"score":0.9,"rationale":null}',
        '{"success":true,"score":0.9,"rationale":"  "}',
        '{"success":true,"score":1e999,"rationale":"target reached"}',
        '[true, 0.9, "target reached"]',
        {"success": True, "score": 0.9, "rationale": "target reached"},
    ],
)
def test_api_judge_rejects_invalid_complete_contract(monkeypatch, content) -> None:
    with pytest.raises(VlmEvalError, match="Hosted VLM response"):
        _call_completion(monkeypatch, _completion(content=content))


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_api_judge_rejects_incomplete_output_even_when_json_valid(
    monkeypatch, finish
) -> None:
    with pytest.raises(VlmEvalError, match="finish_reason=stop"):
        _call_completion(monkeypatch, _completion(finish=finish))


@pytest.mark.parametrize("language", ["json", "JSON", ""])
def test_api_judge_deframes_one_complete_markdown_json_block(
    monkeypatch, language
) -> None:
    content = f"```{language}\n{_VALID_CONTENT}\n```"

    result = _call_completion(monkeypatch, _completion(content=content))

    assert result.score == 0.9
    assert result.evidence is not None
    assert (
        result.evidence.provider.parser_version
        == "npa_vlm_eval_hosted_json_v1+markdown-fence-v1"
    )


def test_api_judge_still_rejects_duplicate_keys_inside_fence(monkeypatch) -> None:
    content = (
        "```json\n"
        '{"success":true,"score":0.1,"score":0.9,"rationale":"target reached"}'
        "\n```"
    )

    with pytest.raises(VlmEvalError, match="duplicate keys"):
        _call_completion(monkeypatch, _completion(content=content))


@pytest.mark.parametrize(
    "content",
    [
        "prefix\n```json\n" + _VALID_CONTENT + "\n```",
        "```json\n" + _VALID_CONTENT + "\n```\nsuffix",
        "```json\n" + _VALID_CONTENT,
    ],
)
def test_api_judge_rejects_partial_or_embedded_fences(monkeypatch, content) -> None:
    with pytest.raises(VlmEvalError, match="could not be parsed in full"):
        _call_completion(monkeypatch, _completion(content=content))


def test_api_judge_requires_completion_metadata(monkeypatch) -> None:
    completion = _completion()
    del completion["choices"][0]["finish_reason"]
    with pytest.raises(VlmEvalError, match="finish_reason=stop"):
        _call_completion(monkeypatch, completion)


@pytest.mark.parametrize("model", [None, "", "  ", 7])
def test_api_judge_requires_actual_model_identity(monkeypatch, model) -> None:
    with pytest.raises(VlmEvalError, match="identify the served model"):
        _call_completion(monkeypatch, _completion(model=model))


@pytest.mark.parametrize(
    "model",
    [
        "MiniMaxAI/MiniMax-M3",
        "google/gemma-3-27b-it",
        "nvidia/Nemotron-3_5-Lightning",
        "openbmb/MiniCPM-V-4_5",
    ],
)
def test_api_judge_rejects_canonical_model_mismatch(monkeypatch, model) -> None:
    with pytest.raises(VlmEvalError, match="does not match"):
        _call_completion(
            monkeypatch, _completion(model="vendor/other-model"), model=model
        )


@pytest.mark.parametrize("score", [0, 0.74, 1])
def test_api_judge_preserves_valid_scores(monkeypatch, score) -> None:
    content = json.dumps(
        {"success": False, "score": score, "rationale": "visible evidence"}
    )
    result = _call_completion(monkeypatch, _completion(content=content))
    assert result.success is False
    assert result.score == score
    assert result.rationale == "visible evidence"


def test_self_hosted_judge_keeps_legacy_parsing_without_completion_metadata(
    monkeypatch,
) -> None:
    completion = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"success":"yes","score":7,"rationale":"legacy"}\n```'
                }
            }
        ]
    }
    result = _call_completion(monkeypatch, completion, backend="self-hosted")
    assert result.success is True
    assert result.score == 1.0
    assert result.served_model is None
    assert result.evidence is not None
    assert (
        result.evidence.provider.parser_version
        == "npa_vlm_eval_compatible_json_v1+markdown-fence-v1"
    )


def test_api_custom_alias_preserves_request_and_reports_actual_judged_model(
    monkeypatch,
    tmp_path,
) -> None:
    from npa.workbench import vlm_eval

    frame = tmp_path / "synthetic.png"
    Image.new("RGB", (8, 8), "green").save(frame)
    requests = []

    def post(**kwargs):
        requests.append(kwargs)
        return _completion(model="vendor/served-vision")

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    result = evaluate_vlm(
        input_path=str(frame),
        output_path=str(tmp_path / "evaluation.json"),
        backend="api",
        model="vendor/explicit-alias",
        endpoint_url="https://example.test/v1",
        task="Judge the green diagram",
    )
    assert requests[0]["request"]["model"] == "vendor/explicit-alias"
    assert requests[0]["url"] == "https://example.test/v1/chat/completions"
    saved = json.loads(json.dumps(asdict(result)))
    assert saved["model"] == "vendor/explicit-alias"
    assert saved["served_model"] == "vendor/served-vision"
    assert saved["score"] == 0.9
    assert saved["passed"] is True
    assert saved["frame_count"] == 1


def test_api_result_retains_recomputable_secret_free_evidence(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    frame = tmp_path / "nested" / "frame.png"
    frame.parent.mkdir()
    Image.new("RGB", (12, 9), "green").save(frame)
    completion = _completion(model="vendor/served-vision")
    completion["id"] = "request-123"
    completion["usage"] = {"prompt_tokens": 31, "completion_tokens": 17}
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **kwargs: completion
    )

    result = evaluate_vlm(
        input_path=str(frame.parent),
        output_path=str(tmp_path / "evaluation.json"),
        backend="api",
        model="vendor/explicit-alias",
        endpoint_url="https://example.test/v1",
        task="Judge the green diagram",
        rubric="Require visible green pixels.",
    )

    evidence = result.evidence
    assert evidence is not None
    assert evidence.schema_version == "npa_vlm_eval_evidence_v1"
    assert evidence.request.endpoint_role == "hosted-api"
    assert len(evidence.request.frames) == 1
    submitted = vlm_eval.select_rollout_frames(frame.parent)[0]
    frame_evidence = evidence.request.frames[0]
    assert frame_evidence.label == "frame.png"
    assert frame_evidence.sha256 == hashlib.sha256(submitted.data).hexdigest()
    assert (frame_evidence.width, frame_evidence.height) == (12, 9)

    manifest_json = vlm_eval._canonical_json(evidence.request.request_manifest)
    assert (
        evidence.request.request_manifest_sha256
        == hashlib.sha256(manifest_json.encode()).hexdigest()
    )
    assert str(frame.parent) not in manifest_json
    assert "example.test" not in manifest_json
    assert "Authorization" not in manifest_json
    assert "data:image" not in manifest_json

    raw_response = vlm_eval._canonical_json(completion)
    assert evidence.provider.provider_request_id == "request-123"
    assert evidence.provider.returned_model == "vendor/served-vision"
    assert evidence.provider.finish_reason == "stop"
    assert evidence.provider.usage == completion["usage"]
    assert evidence.provider.raw_response == raw_response
    assert (
        evidence.provider.raw_response_sha256
        == hashlib.sha256(raw_response.encode()).hexdigest()
    )
    json.dumps(asdict(result))


def test_api_result_marks_unavailable_optional_provider_metadata(monkeypatch) -> None:
    result = _call_completion(monkeypatch, _completion())

    assert result.evidence is not None
    assert result.evidence.provider.provider_request_id is None
    assert result.evidence.provider.returned_model == "MiniMaxAI/MiniMax-M3"
    assert result.evidence.provider.usage is None
    assert result.evidence.provider.status_code is None


def test_api_judge_rejects_explicit_provider_refusal(monkeypatch) -> None:
    completion = _completion()
    completion["choices"][0]["message"]["refusal"] = "I cannot inspect this image."

    with pytest.raises(VlmEvalError, match="refused"):
        _call_completion(monkeypatch, completion)


def test_api_result_retains_exact_http_body_and_header_request_id(monkeypatch) -> None:
    from npa.workbench import vlm_eval

    completion = _completion()
    raw_body = json.dumps(completion, separators=(",", ":")) + "\n"

    class ExactResponse:
        status_code = 200
        headers = {"x-request-id": "header-request-456"}
        text = raw_body

        def raise_for_status(self):
            return None

        def json(self):
            return completion

    class ExactClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return ExactResponse()

    monkeypatch.setattr(vlm_eval.httpx, "Client", ExactClient)
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    result = vlm_eval._call_openai_compatible(
        backend="api",
        model="MiniMaxAI/MiniMax-M3",
        endpoint_url="https://example.test/v1",
        api_key_env="TEST_KEY",
        prompt="Return JSON",
        frames=[],
        timeout_s=120,
    )

    assert result.evidence is not None
    assert result.evidence.provider.provider_request_id == "header-request-456"
    assert result.evidence.provider.status_code == 200
    assert result.evidence.provider.raw_response == raw_body


def test_api_key_falls_back_to_configured_token_factory_credentials(
    monkeypatch,
) -> None:
    from npa.clients import token_factory

    for key in ("VLM_EVAL_API_KEY", "NEBIUS_TOKEN_FACTORY_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        token_factory,
        "resolve_config",
        lambda **kwargs: SimpleNamespace(api_key="configured-key"),
    )

    assert (
        _resolve_api_key(backend="api", api_key_env="VLM_EVAL_API_KEY")
        == "configured-key"
    )


def test_selected_frame_labels_preserve_relative_identity(tmp_path) -> None:
    from npa.workbench.vlm_eval import select_rollout_frames

    root = tmp_path / "rollout"
    for subdirectory, color in (("camera-a", "green"), ("camera-b", "red")):
        path = root / subdirectory / "frame.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(path)

    selected = select_rollout_frames(root, frame_selection="sequence", max_frames=2)

    assert [frame.label for frame in selected] == [
        "camera-a/frame.png",
        "camera-b/frame.png",
    ]


def test_real_benchmark_case_retains_per_request_evidence(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    rollout = tmp_path / "rollout"
    rollout.mkdir()
    Image.new("RGB", (8, 8), "green").save(rollout / "frame.png")
    dataset = tmp_path / "benchmark.json"
    dataset.write_text(
        json.dumps(
            {
                "format": "npa_vlm_eval_benchmark_v1",
                "items": [
                    {
                        "id": "visible-green",
                        "rollout": str(rollout),
                        "expected_label": True,
                        "task": "Confirm that the frame is green.",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval,
        "_post_with_readiness_retry",
        lambda **kwargs: _completion(model="vendor/served-vision"),
    )

    report = benchmark_vlm_eval(
        dataset=str(dataset),
        thresholds=[0.8],
        rubrics=["default"],
        models=["vendor/explicit-alias"],
        backend="api",
    )

    case = report.best_config.results[0]
    assert case.score_source == "api"
    assert case.evidence is not None
    assert case.evidence.request.frames[0].label == "frame.png"
    assert case.evidence.provider.finish_reason == "stop"


def test_benchmark_case_retains_pre_provenance_positional_constructor() -> None:
    from npa.workbench.vlm_eval import VlmBenchmarkCaseResult

    case = VlmBenchmarkCaseResult(
        "case-1",
        "rollout",
        True,
        True,
        0.9,
        "passed",
        True,
        "move the cube",
        "visible completion",
        4,
        "model",
    )
    assert case.evidence is None
    assert case.score_source == "model"
    assert case.frame_count == 4


def _preference_completion(
    preference="B",
    *,
    confidence="high",
    model="MiniMaxAI/MiniMax-M3",
    finish="stop",
):
    content = {
        "preference": preference,
        "confidence": confidence,
        "observable_support": ["visible geometry differs"],
        "critical_defects": {
            "A": ["visible defect in A"],
            "B": ["visible defect in B"],
        },
        "uncertainty": "pixels do not establish physical correctness",
    }
    return {
        "id": f"request-{preference}-{confidence}",
        "model": model,
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": json.dumps(content)},
            }
        ],
    }


def _run_preference_comparison(
    monkeypatch,
    tmp_path,
    completions,
    *,
    task="Compare matched scene views.",
    rubric="Prefer more visible measured detail and fewer unsupported surfaces.",
    model="MiniMaxAI/MiniMax-M3",
):
    from npa.workbench import vlm_eval

    requests = []
    responses = iter(completions)

    def post(**kwargs):
        requests.append(kwargs)
        return next(responses)

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    request = _preference_request(
        tmp_path,
        model=model,
        task=task,
        rubric=rubric,
    )
    report = vlm_eval.compare_vlm_preference(request)
    return report, requests


def _preference_request(
    tmp_path,
    *,
    model="MiniMaxAI/MiniMax-M3",
    task="Compare matched scene views.",
    rubric="Prefer more visible measured detail and fewer unsupported surfaces.",
):
    from npa.workbench import vlm_eval

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (17, 11), "red").save(first)
    Image.new("RGB", (17, 11), "blue").save(second)
    return vlm_eval.VlmPreferenceComparisonRequest(
        baseline_path=str(first),
        candidate_path=str(second),
        output_path=str(tmp_path / "preference"),
        model=model,
        task=task,
        rubric=rubric,
    )


def test_preference_prompt_matches_frozen_contract() -> None:
    from npa.workbench import vlm_eval

    task = "Compare matched views."
    rubric = "Prefer visible measured detail."
    prompt = vlm_eval._preference_prompt(task, rubric)
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == (
        "db3df4e7cbecab7ed359ad5175e5106332799a42955e872c67f1c70417830149"
    )


def _prompt_preference_schema():
    from npa.workbench import vlm_eval

    prompt = vlm_eval._preference_prompt("Compare views.", "Use visible detail.")
    schema = json.loads(
        next(line for line in prompt.splitlines() if line.startswith("{"))
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize("preference", ["A", "B", "tie", "unresolved"])
@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_preference_prompt_schema_accepts_strict_parser_contract(
    preference, confidence
) -> None:
    from npa.workbench import vlm_eval

    completion = _preference_completion(preference, confidence=confidence)
    payload = json.loads(completion["choices"][0]["message"]["content"])
    payload["critical_defects"] = {
        "A": ["No critical defect is visible in this view."],
        "B": ["No critical defect is visible in this view."],
    }

    _prompt_preference_schema().validate(payload)
    verdict = vlm_eval._preference_verdict_from_payload(payload)

    assert verdict.preference == preference
    assert verdict.confidence == confidence
    assert verdict.critical_defects.A == tuple(payload["critical_defects"]["A"])
    assert verdict.critical_defects.B == tuple(payload["critical_defects"]["B"])


def test_preference_prompt_schema_rejects_misspelled_uncertainty() -> None:
    from npa.workbench import vlm_eval

    completion = _preference_completion()
    payload = json.loads(completion["choices"][0]["message"]["content"])
    payload["uncertainity"] = payload.pop("uncertainty")

    assert not _prompt_preference_schema().is_valid(payload)
    with pytest.raises(VlmEvalError, match="fields do not match"):
        vlm_eval._preference_verdict_from_payload(payload)


def _preference_request_image_urls(request):
    return [
        item["image_url"]["url"]
        for item in request["messages"][0]["content"]
        if item["type"] == "image_url"
    ]


def _assert_balanced_preference_requests(report, calls) -> None:
    first, second = (call["request"] for call in calls)
    expected_fields = {"model", "temperature", "messages", "chat_template_kwargs"}
    assert set(first) == set(second) == expected_fields
    assert "max_tokens" not in first and "max_tokens" not in second
    assert first["chat_template_kwargs"] == {"thinking_mode": "disabled"}
    assert "response_format" not in first
    assert first["messages"][0]["content"][0] == second["messages"][0]["content"][0]
    assert first["messages"][0]["content"][1]["text"] == "IMAGE A"
    assert first["messages"][0]["content"][3]["text"] == "IMAGE B"
    first_urls = _preference_request_image_urls(first)
    second_urls = _preference_request_image_urls(second)
    assert first_urls == list(reversed(second_urls))
    submitted = [
        hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
        for url in first_urls
    ]
    assert submitted == [
        report.normalized_baseline_sha256,
        report.normalized_candidate_sha256,
    ]
    request_text = json.dumps([first, second]).lower()
    assert "baseline" not in request_text
    assert "candidate" not in request_text


def _assert_candidate_preference_report(report) -> None:
    assert report.status == "consistent_candidate_preference"
    assert report.mapped_preferences == ("candidate", "candidate")
    assert report.escalation_required is False
    assert report.agreement_eligible is True
    assert report.requests_counterbalanced is True
    assert report.deployment_status == "audit_only"
    assert report.operational_rate_estimated is False
    assert report.first_order.provider is not None
    assert report.first_order.provider.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
    }
    assert report.first_order.request.frames[0].label == "A"
    assert report.reversed_order.request.frames[0].label == "A"
    assert (
        report.first_order.request.frames[0].sha256
        == report.reversed_order.request.frames[1].sha256
    )


def test_compare_preference_balances_orders_and_maps_candidate(
    monkeypatch, tmp_path
) -> None:
    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [_preference_completion("B"), _preference_completion("A")],
    )

    assert len(calls) == 2
    _assert_balanced_preference_requests(report, calls)
    _assert_candidate_preference_report(report)


def test_preference_transport_strips_source_image_metadata(monkeypatch, tmp_path):
    from npa.workbench import vlm_eval

    request = _preference_request(tmp_path)
    for path, arm in (
        (request.baseline_path, "baseline"),
        (request.candidate_path, "candidate"),
    ):
        with Image.open(path) as source:
            exif = Image.Exif()
            exif[270] = f"private {arm} description"
            source.save(path, icc_profile=f"private {arm} profile".encode(), exif=exif)
    requests = []

    def post(**kwargs):
        requests.append(kwargs["request"])
        return _preference_completion("B" if len(requests) == 1 else "A")

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    report = vlm_eval.compare_vlm_preference(request)

    assert len(requests) == 2
    for transport, colors in zip(
        requests, (("red", "blue"), ("blue", "red")), strict=True
    ):
        for url, color in zip(
            _preference_request_image_urls(transport), colors, strict=True
        ):
            with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
                assert image.info == {}
                assert not image.getexif()
                assert image.tobytes() == Image.new("RGB", (17, 11), color).tobytes()
    _assert_candidate_preference_report(report)


def test_sdk_preference_default_rubric_and_journal_finalize_without_transport(
    monkeypatch, tmp_path
):
    from npa.workbench import vlm_eval
    from npa.sdk.workbench.vlm_eval import compare_preference

    assert compare_preference is vlm_eval.compare_vlm_preference
    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [_preference_completion("B"), _preference_completion("A")],
        rubric="",
    )
    assert report.rubric == vlm_eval.DEFAULT_RUBRIC
    destination = Path(report.result_uri)
    assert not destination.exists()
    journal = destination.parent / ".vlm_preference_comparison" / "report-ready.json"
    retained = json.loads(journal.read_text())
    assert retained == json.loads(json.dumps(asdict(report)))

    vlm_eval.write_preference_report(retained, result_uri=report.result_uri)

    assert json.loads(destination.read_text()) == retained
    assert len(calls) == 2
    with pytest.raises(VlmEvalError, match="already exists"):
        vlm_eval.write_preference_report(retained, result_uri=report.result_uri)


def test_kimi_preference_preserves_model_policy_and_reversed_images(
    monkeypatch, tmp_path
):
    model = "moonshotai/Kimi-K3"
    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [
            _preference_completion("B", model=model),
            _preference_completion("A", model=model),
        ],
        model=model,
    )
    _assert_candidate_preference_report(report)
    first, second = (call["request"] for call in calls)
    for request in (first, second):
        assert "temperature" not in request
        assert "max_tokens" not in request
        assert request["reasoning_effort"] == "low"
        assert request["response_format"] == {"type": "json_object"}
    assert _preference_request_image_urls(first) == list(
        reversed(_preference_request_image_urls(second))
    )
    for outcome in (report.first_order, report.reversed_order):
        assert outcome.request.request_manifest["generation_parameters"] == {
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        }


@pytest.mark.parametrize(
    ("first", "second", "confidence", "expected_status", "escalation"),
    [
        ("A", "B", "high", "consistent_baseline_preference", False),
        ("tie", "tie", "high", "consistent_tie", False),
        ("A", "A", "high", "order_disagreement_or_nondeterminism", True),
        ("unresolved", "A", "high", "unresolved", True),
        ("B", "A", "medium", "low_confidence", True),
    ],
)
def test_compare_preference_statuses_fail_closed(
    monkeypatch,
    tmp_path,
    first,
    second,
    confidence,
    expected_status,
    escalation,
) -> None:
    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [
            _preference_completion(first, confidence=confidence),
            _preference_completion(second, confidence=confidence),
        ],
    )

    assert len(calls) == 2
    assert report.status == expected_status
    assert report.escalation_required is escalation
    assert report.agreement_eligible is (not escalation)


def test_compare_preference_runs_second_order_after_first_contract_error(
    monkeypatch, tmp_path
) -> None:
    invalid = _preference_completion()
    invalid["choices"][0]["message"]["content"] = "```json\n{}\n```"
    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [invalid, _preference_completion("A")],
    )

    assert len(calls) == 2
    assert report.status == "judge_error"
    assert report.escalation_required is True
    assert report.first_order.verdict is None
    assert report.first_order.error is not None
    assert report.first_order.error.error_type == "response_contract_error"
    assert report.first_order.provider is not None
    assert report.first_order.provider.raw_response
    assert report.reversed_order.verdict is not None


@pytest.mark.parametrize(
    "mutation",
    [
        {"extra": True},
        {"observable_support": []},
        {"observable_support": [""]},
        {"observable_support": [False]},
        {"critical_defects": {"A": [], "B": ["defect"]}},
        {"critical_defects": {"A": ["defect"], "B": []}},
        {"critical_defects": {"A": ["defect"], "C": ["defect"]}},
        {"critical_defects": {"A": ["defect"], "B": [" "]}},
        {"critical_defects": {"A": ["defect"], "B": "defect"}},
        {"uncertainty": ""},
        {"uncertainty": " "},
        {"uncertainty": None},
        {"uncertainty": "", "uncertainity": ""},
        {"preference": "candidate"},
        {"preference": False},
        {"confidence": "certain"},
    ],
)
def test_preference_parser_rejects_schema_mutations(
    monkeypatch, tmp_path, mutation
) -> None:
    invalid = _preference_completion()
    payload = json.loads(invalid["choices"][0]["message"]["content"])
    payload.update(mutation)
    assert not _prompt_preference_schema().is_valid(payload)
    invalid["choices"][0]["message"]["content"] = json.dumps(payload)

    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [invalid, _preference_completion("A")],
    )

    assert len(calls) == 2
    assert report.status == "judge_error"
    assert report.first_order.error is not None


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{}\n```",
        "prefix {}",
        "{} suffix",
        '{"preference":"A","preference":"B","confidence":"high",'
        '"observable_support":["visible"],'
        '"critical_defects":{"A":["a"],"B":["b"]},"uncertainty":"unknown"}',
        ["not", "a", "string"],
    ],
)
def test_preference_parser_rejects_noncanonical_content(
    monkeypatch, tmp_path, content
) -> None:
    invalid = _preference_completion()
    invalid["choices"][0]["message"]["content"] = content

    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [invalid, _preference_completion("A")],
    )

    assert len(calls) == 2
    assert report.status == "judge_error"
    assert report.first_order.error is not None
    assert report.reversed_order.verdict is not None


@pytest.mark.parametrize(
    ("finish", "model"),
    [
        ("length", "MiniMaxAI/MiniMax-M3"),
        ("content_filter", "MiniMaxAI/MiniMax-M3"),
        ("stop", "vendor/wrong-model"),
    ],
)
def test_preference_parser_rejects_incomplete_or_wrong_model(
    monkeypatch, tmp_path, finish, model
) -> None:
    invalid = _preference_completion(finish=finish, model=model)

    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [invalid, _preference_completion("A")],
    )

    assert len(calls) == 2
    assert report.status == "judge_error"
    assert report.first_order.provider is not None
    assert report.first_order.provider.finish_reason == finish
    assert report.first_order.provider.returned_model == model


def test_preference_parser_rejects_custom_model_mismatch(monkeypatch, tmp_path) -> None:
    report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [
            _preference_completion(model="vendor/other-model"),
            _preference_completion("A", model="vendor/requested-model"),
        ],
        model="vendor/requested-model",
    )

    assert len(calls) == 2
    assert report.status == "judge_error"
    assert report.first_order.error is not None
    assert report.first_order.error.error_type == "response_contract_error"


@pytest.mark.parametrize("refusal", [False, {}, ["no"], "I cannot inspect this image."])
def test_preference_rejects_refusals_and_retains_both_outcomes(
    monkeypatch, tmp_path, refusal
) -> None:
    invalid = _preference_completion("B")
    invalid["choices"][0]["message"]["refusal"] = refusal
    report, calls = _run_preference_comparison(
        monkeypatch, tmp_path, [invalid, _preference_completion("A")]
    )
    assert len(calls) == 2
    assert report.status == "judge_error" and report.escalation_required
    assert report.first_order.verdict is None
    assert report.first_order.error.error_type == "response_contract_error"
    assert report.first_order.provider is not None
    assert report.reversed_order.verdict is not None


def _preference_http_error_responses(vlm_eval):
    return iter(
        [
            (
                vlm_eval._VlmBackendResponse(
                    data={"error": {"message": "rate limited"}},
                    raw_body='{"error":{"message":"rate limited"}}',
                    status_code=429,
                    request_id_header="request-rate-limited",
                    latency_s=0.2,
                ),
                VlmEvalError("provider rejected request"),
            ),
            (
                vlm_eval._coerce_backend_response(
                    _preference_completion("A"),
                    fallback_latency_s=0.3,
                ),
                None,
            ),
        ]
    )


def test_compare_preference_retains_http_error_and_runs_reversed_order(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    responses = _preference_http_error_responses(vlm_eval)
    calls = []

    def post(**_kwargs):
        calls.append(True)
        return next(responses)

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    report = vlm_eval.compare_vlm_preference(_preference_request(tmp_path))

    assert len(calls) == 2
    assert report.status == "judge_error"
    assert report.first_order.error is not None
    assert report.first_order.error.stage == "provider_http_status"
    assert report.first_order.provider is not None
    assert report.first_order.provider.status_code == 429
    assert report.first_order.provider.provider_request_id == "request-rate-limited"
    assert report.first_order.provider.raw_response == (
        '{"error":{"message":"rate limited"}}'
    )
    assert report.reversed_order.verdict is not None


def test_compare_preference_crash_journal_blocks_duplicate_transport(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    request = _preference_request(tmp_path)
    output = Path(request.output_path)
    calls = []

    def crash(**_kwargs):
        calls.append(True)
        request_path = output / ".vlm_preference_comparison" / "request-01.json"
        assert request_path.exists()
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", crash)

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        vlm_eval.compare_vlm_preference(request)
    assert len(calls) == 1
    journal = output / ".vlm_preference_comparison"
    assert (journal / "state.json").exists()
    assert (journal / "request-01.json").exists()
    assert not (journal / "response-01.json").exists()
    assert journal.stat().st_mode & 0o777 == 0o700
    assert (journal / "request-01.json").stat().st_mode & 0o777 == 0o600

    with pytest.raises(VlmEvalError, match="already exists"):
        vlm_eval.compare_vlm_preference(request)
    assert len(calls) == 1


def test_compare_preference_journals_response_at_transport_boundary(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    request = _preference_request(tmp_path)
    output = Path(request.output_path)
    raw_body = '{"id":"private-id","choices":[]}'

    def crash(*, response_sink, **_kwargs):
        response_sink(
            vlm_eval._VlmBackendResponse(
                data={},
                raw_body=raw_body,
                status_code=200,
                request_id_header="private-header-id",
                latency_s=0.25,
            )
        )
        raise RuntimeError("simulated interruption after transport")

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", crash)

    with pytest.raises(RuntimeError, match="after transport"):
        vlm_eval.compare_vlm_preference(request)

    journal = output / ".vlm_preference_comparison"
    boundary = json.loads(
        (journal / "transport-boundary-01.json").read_text(encoding="utf-8")
    )
    assert boundary["raw_body"] == raw_body
    assert boundary["request_id_header"] == "private-header-id"
    assert not (journal / "response-01.json").exists()
    assert (journal / "transport-boundary-01.json").stat().st_mode & 0o777 == 0o600


def test_preference_transport_aborts_when_response_journal_fails(monkeypatch) -> None:
    from npa.workbench import vlm_eval

    response = vlm_eval._coerce_backend_response(
        _preference_completion(),
        fallback_latency_s=0.1,
    )

    def post(*, response_sink, **_kwargs):
        response_sink(response)
        return response

    def fail_retention(_response):
        raise VlmEvalError("simulated journal failure")

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    with pytest.raises(VlmEvalError, match="could not be retained"):
        vlm_eval._post_comparison_request(
            url="https://provider.invalid/v1/chat/completions",
            headers={},
            request={"model": "test"},
            timeout_s=1,
            response_sink=fail_retention,
        )


def test_compare_preference_rejects_source_role_text_before_transport(
    monkeypatch, tmp_path
) -> None:
    called = False

    def post(**_kwargs):
        nonlocal called
        called = True

    from npa.workbench import vlm_eval

    baseline = tmp_path / "first.png"
    candidate = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(baseline)
    Image.new("RGB", (8, 8), "blue").save(candidate)
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)

    with pytest.raises(VlmEvalError, match="source-role words"):
        vlm_eval.compare_vlm_preference(
            vlm_eval.VlmPreferenceComparisonRequest(
                baseline_path=str(baseline),
                candidate_path=str(candidate),
                output_path=str(tmp_path / "preference"),
                task="Prefer candidate_output over baseline_output.",
                rubric="Use visible detail.",
            )
        )
    assert called is False


def test_compare_preference_rejects_source_role_substring_in_model(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    baseline = tmp_path / "first.png"
    candidate = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(baseline)
    Image.new("RGB", (8, 8), "blue").save(candidate)
    called = False

    def post(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    with pytest.raises(VlmEvalError, match="model ID"):
        vlm_eval.compare_vlm_preference(
            vlm_eval.VlmPreferenceComparisonRequest(
                baseline_path=str(baseline),
                candidate_path=str(candidate),
                output_path=str(tmp_path / "preference"),
                model="org/candidate_model",
                task="Compare matched views.",
                rubric="Use visible detail.",
            )
        )
    assert called is False


def test_compare_preference_rejects_api_key_value_as_environment_name(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    baseline = tmp_path / "first.png"
    candidate = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(baseline)
    Image.new("RGB", (8, 8), "blue").save(candidate)
    monkeypatch.setattr(
        vlm_eval,
        "_post_with_readiness_retry",
        lambda **_kwargs: pytest.fail("transport must not run"),
    )

    with pytest.raises(VlmEvalError, match="environment variable name"):
        vlm_eval.compare_vlm_preference(
            vlm_eval.VlmPreferenceComparisonRequest(
                baseline_path=str(baseline),
                candidate_path=str(candidate),
                output_path=str(tmp_path / "preference"),
                api_key_env="secret-key-value.with-punctuation",
                task="Compare matched views.",
                rubric="Use visible detail.",
            )
        )


def test_compare_preference_requires_exactly_one_image_per_input(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    baseline = tmp_path / "first"
    baseline.mkdir()
    Image.new("RGB", (8, 8), "red").save(baseline / "one.png")
    Image.new("RGB", (8, 8), "green").save(baseline / "two.png")
    candidate = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "blue").save(candidate)
    called = False

    def post(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    with pytest.raises(VlmEvalError, match="exactly one"):
        vlm_eval.compare_vlm_preference(
            vlm_eval.VlmPreferenceComparisonRequest(
                baseline_path=str(baseline),
                candidate_path=str(candidate),
                output_path=str(tmp_path / "preference"),
                task="Compare matched views.",
                rubric="Prefer visible detail.",
            )
        )
    assert called is False


def test_compare_preference_ignores_ambient_endpoint_override(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("VLM_EVAL_API_BASE_URL", "https://ambient.invalid/v1")
    _report, calls = _run_preference_comparison(
        monkeypatch,
        tmp_path,
        [_preference_completion("B"), _preference_completion("A")],
    )

    assert {call["url"] for call in calls} == {
        "https://api.tokenfactory.nebius.com/v1/chat/completions"
    }


def test_compare_preference_requires_canonical_new_destination(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    canonical = str(tmp_path / "vlm_preference_comparison.json")
    assert vlm_eval.preference_comparison_result_uri_for(canonical) == canonical
    assert vlm_eval.preference_comparison_result_uri_for(
        str(tmp_path / "evidence")
    ) == str(tmp_path / "evidence" / "vlm_preference_comparison.json")
    with pytest.raises(VlmEvalError, match="filename must be"):
        vlm_eval.preference_comparison_result_uri_for(str(tmp_path / "wrong.json"))

    canonical_path = tmp_path / "vlm_preference_comparison.json"
    canonical_path.write_text("{}\n", encoding="utf-8")
    called = False

    def post(**_kwargs):
        nonlocal called
        called = True

    baseline = tmp_path / "first.png"
    candidate = tmp_path / "second.png"
    Image.new("RGB", (8, 8), "red").save(baseline)
    Image.new("RGB", (8, 8), "blue").save(candidate)
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    with pytest.raises(VlmEvalError, match="already exists"):
        vlm_eval.compare_vlm_preference(
            vlm_eval.VlmPreferenceComparisonRequest(
                baseline_path=str(baseline),
                candidate_path=str(candidate),
                output_path=canonical,
                task="Compare matched views.",
                rubric="Use visible detail.",
            )
        )
    assert called is False


def test_write_preference_report_is_private_and_no_clobber(tmp_path) -> None:
    from npa.workbench import vlm_eval

    output = tmp_path / "private" / "vlm_preference_comparison.json"
    written = vlm_eval.write_preference_report(
        {"status": "judge_error"},
        result_uri=str(output),
    )

    assert written == str(output)
    assert output.read_text(encoding="utf-8")
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(VlmEvalError, match="already exists"):
        vlm_eval.write_preference_report(
            {"status": "replacement"},
            result_uri=str(output),
        )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "judge_error"


def test_write_preference_report_uses_atomic_object_create() -> None:
    from npa.workbench import vlm_eval

    calls = []

    class Storage:
        def put_bytes_conditional(self, payload, uri, **kwargs):
            calls.append((payload, uri, kwargs))
            return '"etag"'

    uri = "s3://private-role/evidence/vlm_preference_comparison.json"
    assert (
        vlm_eval.write_preference_report(
            {"status": "consistent_tie"},
            result_uri=uri,
            storage_client=Storage(),
        )
        == uri
    )
    assert len(calls) == 1
    payload, written_uri, kwargs = calls[0]
    assert json.loads(payload)["status"] == "consistent_tie"
    assert written_uri == uri
    assert kwargs == {
        "if_none_match": True,
        "content_type": "application/json",
    }
