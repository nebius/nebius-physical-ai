"""Check terminal-evidence instructions at the transport and score-gate boundaries."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy

from PIL import Image
import pytest

from npa.workbench import vlm_eval


@pytest.fixture(autouse=True)
def _synthetic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_EVAL_API_KEY", "unit-test-key")


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
@pytest.mark.parametrize("outcome", ["missing", "ambiguous"])
def test_terminal_failure_instruction_reaches_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str, outcome: str
) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), "gray").save(frame)
    requests = []

    def respond(**kwargs):
        requests.append(kwargs["request"])
        return _response(
            score=0.0, success=False, rationale=f"{outcome} terminal state"
        )

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", respond)
    result = vlm_eval.evaluate_vlm(
        input_path=str(frame),
        output_path=str(tmp_path / "result.json"),
        backend=backend,
        model="openbmb/MiniCPM-V-4_5",
        success_threshold=0.8,
    )
    prompt = requests[0]["messages"][0]["content"][0]["text"]
    assert f"Rubric: {vlm_eval.DEFAULT_RUBRIC}" in prompt
    assert "Assign score 0.0 and success false" in prompt
    assert "terminal state is missing or ambiguous" in prompt
    assert result.passed is False and result.status == "needs_iteration"
    assert result.score == 0.0
    raw = json.loads(result.evidence.provider.raw_response)
    assert json.loads(raw["choices"][0]["message"]["content"])["success"] is False
    assert (
        result.evidence.request.rubric_sha256
        == hashlib.sha256(vlm_eval.DEFAULT_RUBRIC.encode()).hexdigest()
    )
    vlm_eval.write_result(asdict(result), result_uri=result.result_uri)
    assert json.loads(Path(result.result_uri).read_text())["passed"] is False


def _response(*, score: float, success: bool, rationale: str) -> dict:
    return {
        "model": "openbmb/MiniCPM-V-4_5",
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "score": score,
                            "success": success,
                            "rationale": rationale,
                        }
                    )
                },
            }
        ],
    }


def test_custom_rubric_and_score_gate_remain_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), "gray").save(frame)
    requests = []

    def respond(**kwargs):
        requests.append(kwargs["request"])
        return _response(score=0.9, success=False, rationale="missing terminal state")

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", respond)
    result = vlm_eval.evaluate_vlm(
        input_path=str(frame),
        output_path=str(tmp_path / "result.json"),
        backend="api",
        model="openbmb/MiniCPM-V-4_5",
        rubric="Score visible progress.",
    )
    prompt = requests[0]["messages"][0]["content"][0]["text"]
    assert "Rubric: Score visible progress." in prompt
    assert "Assign score 0.0" not in prompt
    assert result.passed is True
    raw = json.loads(result.evidence.provider.raw_response)
    assert json.loads(raw["choices"][0]["message"]["content"])["success"] is False


def test_sample_default_rubric_matches_runtime_instruction() -> None:
    dataset = vlm_eval.load_benchmark_dataset(
        str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH)
    )
    assert dataset.rubrics["default"] == vlm_eval.DEFAULT_RUBRIC


def _live_lane() -> dict:
    path = Path(__file__).parents[1] / "e2e/test_vlm_terminal_evidence_live.py"
    return runpy.run_path(str(path))


def test_explicit_live_lane_fails_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    monkeypatch.delenv("NPA_VLM_TERMINAL_LIVE_CONFIG", raising=False)
    with pytest.raises(
        AssertionError, match="NPA_VLM_TERMINAL_LIVE_CONFIG is required"
    ):
        _live_lane()["_load_control_config"]()


def _write_live_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides
) -> dict:
    config = {
        "model": "openbmb/MiniCPM-V-4_5",
        "task": "Judge visible completion.",
        "max_frames": 3,
        "success_threshold": 0.8,
        "output_dir": str(tmp_path / "outputs"),
        "cases": {name: {} for name in _live_lane()["CASES"]},
        **overrides,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    monkeypatch.setenv("NPA_VLM_TERMINAL_LIVE_CONFIG", str(path))
    return config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_frames", True),
        ("max_frames", None),
        ("max_frames", "2"),
        ("max_frames", 2.0),
        ("max_frames", 1),
        ("success_threshold", True),
        ("success_threshold", None),
        ("success_threshold", "0.8"),
        ("success_threshold", float("nan")),
        ("success_threshold", float("inf")),
        ("success_threshold", -0.1),
        ("success_threshold", 0),
        ("success_threshold", 1.1),
    ],
)
def test_live_config_rejects_nonliteral_or_out_of_domain_scalars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: object
) -> None:
    _write_live_config(monkeypatch, tmp_path, **{field: value})
    with pytest.raises((ValueError, AssertionError), match=field):
        _live_lane()["_load_control_config"]()
    assert not (tmp_path / "outputs").exists()


def test_live_config_preserves_valid_literal_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _write_live_config(monkeypatch, tmp_path)
    assert _live_lane()["_load_control_config"]() == config


def _synthetic_controls(tmp_path: Path) -> dict:
    colors = {
        "complete": ["red", "green", "blue"],
        "missing-terminal": ["red", "green"],
        "ambiguous-terminal": ["red", "gray"],
        "no-evidence": ["gray", "gray"],
    }
    cases = {}
    for case, sequence in colors.items():
        directory = tmp_path / case
        directory.mkdir()
        for index, color in enumerate(sequence):
            Image.new("RGB", (32, 32), color).save(directory / f"{index}.png")
        frames = vlm_eval.select_rollout_frames(
            directory, frame_selection="sequence", max_frames=3
        )
        cases[case] = {
            "input_path": str(directory),
            "frame_sha256": [
                hashlib.sha256(frame.data).hexdigest() for frame in frames
            ],
        }
    return {"max_frames": 3, "cases": cases}


def test_live_controls_reject_changed_pixels_and_unmatched_truncation(
    tmp_path: Path,
) -> None:
    config = _synthetic_controls(tmp_path)
    verify = _live_lane()["_verify_frozen_frames"]
    verify(config)
    config["cases"]["missing-terminal"] = config["cases"]["ambiguous-terminal"]
    with pytest.raises(AssertionError, match="Truncate the same source sequence"):
        verify(config)
    Image.new("RGB", (32, 32), "yellow").save(tmp_path / "complete/0.png")
    with pytest.raises(AssertionError, match="Submitted pixels differ"):
        verify(config)


@pytest.mark.parametrize("score", [0.00001, 0.2, 0.95])
def test_live_lane_rejects_nonzero_terminal_control_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, score: float
) -> None:
    config = _synthetic_controls(tmp_path)
    config.update(
        model="openbmb/MiniCPM-V-4_5",
        task="Judge visible completion.",
        success_threshold=0.8,
    )
    response = _response(score=score, success=False, rationale="missing terminal state")
    monkeypatch.setattr(
        vlm_eval,
        "_post_with_readiness_retry",
        lambda **kwargs: vlm_eval._VlmBackendResponse(
            response, json.dumps(response), 200, None, 0.1
        ),
    )
    lane = _live_lane()
    result = lane["_record_control"](config, "missing-terminal", tmp_path)
    saved = tmp_path / "missing-terminal.json"
    assert json.loads(saved.read_text())["score"] == round(score, 4)
    assert saved.stat().st_mode & 0o777 == 0o600
    with pytest.raises(
        AssertionError, match="Frozen control failed|ignored terminal-evidence rubric"
    ):
        lane["_assert_control_result"](config, "missing-terminal", result)
