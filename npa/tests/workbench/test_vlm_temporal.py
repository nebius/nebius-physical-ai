"""Exercise the real Token Factory request path and strict timestamped visual-evidence contract."""

from copy import deepcopy
import json

import httpx
import numpy as np
import pytest

from npa.clients.token_factory import TokenFactoryClient, TokenFactoryConfig
from npa.workbench.vlm_eval.temporal import _validate_response, judge_manipulation, sample_indices

MODEL = "MiniMaxAI/MiniMax-M3"


def _response():
    verdict = {
        "lifted": {"verdict": "yes", "frames": [8, 9], "explanation": "Target above the table in the gripper."},
        "held_at_end": {"verdict": "yes", "frames": [8, 9], "explanation": "Target remains in the gripper."},
        "scene_disturbed": {"verdict": "no", "frames": [0, 9], "explanation": "Fixture and distractors unchanged."},
        "failure_modes": ["none"], "rationale": "Visible lift and final hold.",
    }
    return {"id": "request-fixture", "model": MODEL, "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}]}


def test_sampling_includes_start_end_and_dense_final_window():
    indices = sample_indices(250, 50, 16)
    assert indices[0] == 0 and indices[-1] == 249
    assert len(indices) <= 16 and len(set(indices)) == len(indices)
    assert sum(index >= 224 for index in indices) >= 8


@pytest.mark.parametrize("change", ["model", "truncated", "missing", "nonexistent_frame", "duplicate_frame",
                                  "single_frame_lift", "hold_without_end", "hold_without_lift", "contradictory_tags"])
def test_invalid_or_unsupported_visual_claims_are_rejected(change):
    response = deepcopy(_response())
    value = json.loads(response["choices"][0]["message"]["content"])
    if change == "model":
        response["model"] = "substituted"
    elif change == "truncated":
        response["choices"][0]["finish_reason"] = "length"
    elif change == "missing":
        del value["lifted"]
    elif change in {"nonexistent_frame", "duplicate_frame", "single_frame_lift"}:
        value["lifted"]["frames"] = {"nonexistent_frame": [8, 100], "duplicate_frame": [8, 8],
                                     "single_frame_lift": [8]}[change]
    elif change == "hold_without_end":
        value["held_at_end"]["frames"] = [0, 8]
    elif change == "hold_without_lift":
        value["lifted"]["verdict"] = "no"
    else:
        value["failure_modes"] = ["none", "slip_or_drop"]
    response["choices"][0]["message"]["content"] = json.dumps(value)
    with pytest.raises(ValueError):
        _validate_response(response, MODEL, [{"index": index} for index in (0, 8, 9)])


def test_real_client_sends_blinded_pixels_and_retains_provider_accounting(tmp_path):
    captured = []

    def serve(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    pixels = np.zeros((10, 16, 16, 3), dtype=np.uint8)
    pixels[:, 0, 0] = np.arange(10)[:, None]
    source = tmp_path / "rgb.npy"
    np.save(source, pixels)
    config = TokenFactoryConfig("https://example.invalid/v1/", "test-credential")
    client = TokenFactoryClient(config, http_client=httpx.Client(transport=httpx.MockTransport(serve)))
    result = judge_manipulation(rgb_path=source, output=tmp_path / "judgment", task="Lift the orange spool",
                                fps=2, frame_count=16, model=MODEL, client=client)
    assert len(captured) == 1 and captured[0]["temperature"] == 0
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 50}
    assert result["cost"] is None and result["request_id"] == "request-fixture"
    assert result["transport"]["attempts"] == 1
    content = captured[0]["messages"][0]["content"]
    assert sum(item["type"] == "image_url" for item in content) == len(result["frames"])
    prompt = json.loads((tmp_path / "judgment/request.json").read_text())["prompt"]
    assert "checkpoint_sha256" not in prompt and "reference_lifted" not in prompt
    assert (tmp_path / "judgment/response.json").is_file()


@pytest.mark.parametrize("mutation", ["hold_without_lift", "invalid_json", "empty_choices", "message_null", "choices_scalar"])
def test_invalid_response_is_retained_and_replayed_without_a_second_request(tmp_path, mutation):
    response = _response()
    if mutation == "hold_without_lift":
        verdict = json.loads(response["choices"][0]["message"]["content"])
        verdict["lifted"]["verdict"] = "no"
        response["choices"][0]["message"]["content"] = json.dumps(verdict)
    elif mutation == "invalid_json":
        response["choices"][0]["message"]["content"] = "not JSON"
    elif mutation == "empty_choices":
        response["choices"] = []
    elif mutation == "message_null":
        response["choices"][0]["message"] = None
    else:
        response["choices"] = 7
    calls = []

    def serve(request):
        calls.append(request)
        return httpx.Response(200, json=response)

    client = TokenFactoryClient(TokenFactoryConfig("https://example.invalid/v1/", "test-credential"),
        http_client=httpx.Client(transport=httpx.MockTransport(serve)))
    source = tmp_path / "rgb.npy"
    np.save(source, np.zeros((10, 16, 16, 3), dtype=np.uint8))
    options = dict(rgb_path=source, task="Lift the orange spool", fps=2, frame_count=16, model=MODEL, client=client)
    first = judge_manipulation(output=tmp_path / "first", **options)
    assert first["status"] == "invalid_response" and first["verdict"] is None
    assert first["validation_error"] and first["usage"] == response["usage"]
    replay = judge_manipulation(output=tmp_path / "replay", previous=tmp_path / "first", **options)
    assert replay["status"] == "invalid_response" and replay["verdict"] is None
    assert replay["response_source"] == "replayed" and replay["response_sha256"] == first["response_sha256"]
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["prompt", "frame", "pixels"])
def test_recorded_judgment_cannot_be_reused_for_changed_input(tmp_path, mutation):
    client = TokenFactoryClient(TokenFactoryConfig("https://example.invalid/v1/", "test-credential"),
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_response()))))
    source = tmp_path / "rgb.npy"
    np.save(source, np.zeros((10, 16, 16, 3), dtype=np.uint8))
    options = dict(rgb_path=source, task="Lift the orange spool", fps=2, frame_count=16, model=MODEL, client=client)
    judge_manipulation(output=tmp_path / "first", **options)
    if mutation == "prompt":
        options["task"] = "Lift the blue bottle"
    elif mutation == "frame":
        (tmp_path / "first/frame-000000.jpg").write_bytes(b"changed")
    else:
        np.save(source, np.ones((10, 16, 16, 3), dtype=np.uint8) * 255)
    with pytest.raises(ValueError, match="differs|differ"):
        judge_manipulation(output=tmp_path / "replay", previous=tmp_path / "first", **options)
