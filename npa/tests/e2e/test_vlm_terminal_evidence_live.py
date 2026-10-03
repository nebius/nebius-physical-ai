"""Judge frozen real rollout controls through the hosted VLM with private evidence.

The runbook defines NPA_VLM_TERMINAL_LIVE_CONFIG. Explicit integration runs fail
on missing configuration; the ordinary offline suite skips via the e2e marker.
Operator labels and source pixels must be reviewed before running this test.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.literal_values import require_integer, require_number
from npa.workbench import vlm_eval

pytestmark = [pytest.mark.e2e, pytest.mark.token_factory_e2e]
CASES = ("complete", "missing-terminal", "ambiguous-terminal", "no-evidence")


def _load_control_config() -> dict:
    assert os.environ.get("NPA_INTEGRATION_E2E") == "1", (
        "Enable this live lane explicitly"
    )
    configured = os.environ.get("NPA_VLM_TERMINAL_LIVE_CONFIG", "")
    assert configured, "NPA_VLM_TERMINAL_LIVE_CONFIG is required for this live lane"
    path = Path(configured)
    assert path.is_file() and path.stat().st_mode & 0o077 == 0
    config = json.loads(path.read_text())
    assert set(config) == {
        "model",
        "task",
        "max_frames",
        "success_threshold",
        "output_dir",
        "cases",
    }
    assert isinstance(config["model"], str) and config["model"].strip()
    assert isinstance(config["task"], str) and config["task"].strip()
    require_integer(config["max_frames"], field="max_frames", minimum=2)
    threshold = require_number(
        config["success_threshold"], field="success_threshold", minimum=0, maximum=1
    )
    assert threshold > 0, "success_threshold must be positive"
    assert set(config["cases"]) == set(CASES)
    return config


def _verify_frozen_frames(config: dict) -> None:
    for case in CASES:
        control = config["cases"][case]
        assert set(control) == {"input_path", "frame_sha256"}
        assert Path(control["input_path"]).is_dir()
        frames = vlm_eval.select_rollout_frames(
            control["input_path"],
            frame_selection="sequence",
            max_frames=config["max_frames"],
        )
        assert len(frames) >= 2
        source_files = [
            path
            for path in Path(control["input_path"]).rglob("*")
            if path.is_file() and path.suffix.lower() in vlm_eval.IMAGE_SUFFIXES
        ]
        assert len(frames) == len(source_files), "Freeze every selected frame"
        assert [hashlib.sha256(frame.data).hexdigest() for frame in frames] == (
            control["frame_sha256"]
        ), "Submitted pixels differ from the reviewed control"
    complete = config["cases"]["complete"]["frame_sha256"]
    truncated = config["cases"]["missing-terminal"]["frame_sha256"]
    assert len(truncated) < len(complete)
    assert truncated == complete[: len(truncated)], "Truncate the same source sequence"
    assert len(set(complete)) > 1
    assert config["cases"]["ambiguous-terminal"]["frame_sha256"] != complete
    blank = config["cases"]["no-evidence"]["frame_sha256"]
    assert len(set(blank)) == 1 and blank[0] not in complete


def _record_control(config: dict, case: str, output: Path) -> vlm_eval.VlmEvalResult:
    destination = output / f"{case}.json"
    result = vlm_eval.evaluate_vlm(
        input_path=config["cases"][case]["input_path"],
        output_path=str(destination),
        task=config["task"],
        backend="api",
        model=config["model"],
        success_threshold=config["success_threshold"],
        frame_selection="sequence",
        max_frames=config["max_frames"],
        rubric=vlm_eval.DEFAULT_RUBRIC,
    )
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(asdict(result), stream, indent=2)
    return result


def _assert_control_result(
    config: dict, case: str, result: vlm_eval.VlmEvalResult
) -> None:
    assert result.backend == "api" and result.evidence is not None
    assert result.served_model == config["model"]
    request = result.evidence.request
    assert [frame.sha256 for frame in request.frames] == (
        config["cases"][case]["frame_sha256"]
    )
    assert (
        request.rubric_sha256
        == hashlib.sha256(vlm_eval.DEFAULT_RUBRIC.encode()).hexdigest()
    )
    provider = result.evidence.provider
    assert provider.status_code == 200 and provider.finish_reason == "stop"
    assert (
        provider.raw_response_sha256
        == hashlib.sha256(provider.raw_response.encode()).hexdigest()
    )
    raw = json.loads(provider.raw_response)
    assert raw["model"] == config["model"] and raw.get("usage")
    verdict = vlm_eval._parse_api_structured_response(
        raw["choices"][0]["message"]["content"], served_model=config["model"]
    )
    assert result.passed is (case == "complete"), f"Frozen control failed: {case}"
    if case != "complete":
        assert verdict.score == 0.0, f"Judge ignored terminal-evidence rubric: {case}"
        assert verdict.success is False


def test_real_rollouts_require_visible_terminal_evidence() -> None:
    config = _load_control_config()
    _verify_frozen_frames(config)
    output = Path(config["output_dir"])
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    results = {case: _record_control(config, case, output) for case in CASES}
    for case, result in results.items():
        _assert_control_result(config, case, result)
