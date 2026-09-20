from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
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
    assert result.provider_success is None
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
    assert evidence.schema_version == "npa_vlm_eval_evidence_v2"
    assert evidence.request.endpoint_role == "hosted-api"
    assert len(evidence.request.frames) == 1
    assert result.rubric == "Require visible green pixels."
    reconstructed_prompt = vlm_eval._build_prompt(
        task=result.task,
        rubric=result.rubric,
        frame_selection=result.frame_selection,
        frame_count=result.frame_count,
    )
    assert hashlib.sha256(reconstructed_prompt.encode()).hexdigest() == (
        evidence.request.prompt_sha256
    )
    submitted = vlm_eval.select_rollout_frames(frame.parent)[0]
    frame_evidence = evidence.request.frames[0]
    assert frame_evidence.label == "frame.png"
    assert frame_evidence.sha256 == hashlib.sha256(submitted.data).hexdigest()
    assert (frame_evidence.width, frame_evidence.height) == (12, 9)
    assert frame_evidence.source_kind == "image-sequence"
    assert frame_evidence.source_index == 0
    assert frame_evidence.source_count == 1
    assert frame_evidence.source_timestamp_s is None
    assert evidence.request.request_manifest["sampling"] == {
        "strategy": "keyframes",
        "max_frames": 4,
        "selected_count": 1,
        "source_kind": "image-sequence",
        "source_count": 1,
        "selected_indices": [0],
        "selected_timestamps_s": [None],
        "coverage_complete": True,
        "timestamps_complete": None,
    }

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
    assert result.provider_success is True
    assert result.provider_success_matches_score_gate is True
    assert evidence.provider.raw_response == raw_response
    assert (
        evidence.provider.raw_response_sha256
        == hashlib.sha256(raw_response.encode()).hexdigest()
    )
    json.dumps(asdict(result))


def test_api_result_retains_image_sequence_sampling_coverage(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    rollout = tmp_path / "rollout"
    rollout.mkdir()
    for index in range(5):
        Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(
            rollout / f"frame-{index:03d}.png"
        )
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval,
        "_post_with_readiness_retry",
        lambda **kwargs: _completion(model="vendor/served-vision"),
    )

    result = evaluate_vlm(
        input_path=str(rollout),
        output_path=str(tmp_path / "evaluation.json"),
        backend="api",
        model="vendor/explicit-alias",
        endpoint_url="https://example.test/v1",
        task="Identify visible sequence changes.",
        frame_selection="keyframes",
        max_frames=3,
    )

    assert result.evidence is not None
    frames = result.evidence.request.frames
    assert [frame.source_index for frame in frames] == [0, 2, 4]
    assert [frame.source_count for frame in frames] == [5, 5, 5]
    sampling = result.evidence.request.request_manifest["sampling"]
    assert sampling["selected_indices"] == [0, 2, 4]
    assert sampling["source_count"] == 5
    assert sampling["selected_count"] == 3
    assert sampling["max_frames"] == 3
    assert sampling["coverage_complete"] is True


def test_api_result_surfaces_provider_success_score_contradiction(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    frame = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), "gray").save(frame)
    completion = _completion(
        content='{"success":true,"score":0.0,"rationale":"No task evidence is visible."}'
    )
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **kwargs: completion
    )

    result = evaluate_vlm(
        input_path=str(frame),
        output_path=str(tmp_path / "evaluation.json"),
        backend="api",
        model="MiniMaxAI/MiniMax-M3",
        task="Confirm visible task completion.",
        success_threshold=0.8,
    )

    assert result.score == 0.0
    assert result.passed is False
    assert result.status == "needs_iteration"
    assert result.provider_success is True
    assert result.provider_success_matches_score_gate is False


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
    assert case.provider_success is True
    assert case.provider_success_matches_score_gate is True
