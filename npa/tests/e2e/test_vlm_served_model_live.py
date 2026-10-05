"""Verify model and sampling provenance against an operator-owned GPU endpoint."""

import hashlib
import json
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import select_rollout_frames
from .vlm_sampling_live_helpers import (
    assert_sampling_evidence,
    make_sampling_input,
    validate_sampling_scalars,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.fixture(scope="module")
def live_config() -> dict:
    """Read the private endpoint configuration; inference never uses a mock."""
    config_path = os.environ.get("NPA_VLM_PROVENANCE_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not config_path:
        pytest.skip("provide an operator-owned NPA_VLM_PROVENANCE_LIVE_CONFIG")
    config = json.loads(Path(config_path).read_text())
    assert Path(config["input_path"]).exists(), (
        "Use local input for payload verification"
    )
    assert Path(config["output_path"]).suffix == ".json", "Use a local JSON output file"
    return config


def _run_evaluation(config: dict, *, strategy: str, max_frames: int) -> dict:
    options = {
        "input-path": config["input_path"],
        "output-path": config["output_path"],
        "task": config["task"],
        "backend": "self-hosted",
        "endpoint-url": config["endpoint_url"],
        "model": config["model"],
        "api-key-env": config.get("api_key_env", "VLM_EVAL_API_KEY"),
        "frame-selection": strategy,
        "max-frames": str(max_frames),
        "success-threshold": "0.8",
        "output": "json",
    }
    args = ["workbench", "vlm-eval", "run"]
    for name, value in options.items():
        args.extend([f"--{name}", value])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    validate_sampling_scalars(payload)
    _assert_result_evidence(payload, config)
    _assert_self_hosted_judge_claims(payload)
    assert payload.pop("written_uri") == config["output_path"]
    assert json.loads(Path(config["output_path"]).read_text()) == payload
    return payload


def _assert_result_evidence(payload: dict, config: dict) -> None:
    assert payload["served_model"] == config["expected_served_model"]
    assert payload["model"] == config["model"]
    assert 0 <= payload["score"] <= 1
    assert payload["backend"] == "self-hosted" and not payload["dry_run"]
    evidence = payload["evidence"]
    assert evidence["schema_version"] == "npa_vlm_eval_evidence_v2"
    assert evidence["request"]["endpoint_role"] == "self-hosted"
    manifest = json.dumps(
        evidence["request"]["request_manifest"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert (
        evidence["request"]["request_manifest_sha256"]
        == hashlib.sha256(manifest.encode()).hexdigest()
    )
    _assert_provider_evidence(evidence["provider"], config)


def test_self_hosted_result_retains_served_model(live_config: dict) -> None:
    payload = _run_evaluation(live_config, strategy="keyframes", max_frames=8)
    request = payload["evidence"]["request"]
    selected = select_rollout_frames(
        live_config["input_path"], frame_selection="keyframes", max_frames=8
    )
    assert request["frames"] == request["request_manifest"]["frames"]
    assert [frame["sha256"] for frame in request["frames"]] == [
        hashlib.sha256(frame.data).hexdigest() for frame in selected
    ]
    sampling = request["request_manifest"]["sampling"]
    assert sampling["strategy"] == "keyframes"
    assert sampling["max_frames"] == 8
    assert sampling["selected_count"] == payload["frame_count"] == len(selected)
    assert sampling["selected_indices"] == [frame.source_index for frame in selected]
    assert sampling["selected_timestamps_s"] == [
        frame.source_timestamp_s for frame in selected
    ]


@pytest.mark.parametrize("kind", ["image-sequence", "numpy-episode", "video"])
@pytest.mark.parametrize("strategy", ["final", "keyframes", "sequence"])
def test_self_hosted_sampling_binds_source_payloads(
    live_config: dict, tmp_path: Path, kind: str, strategy: str
) -> None:
    input_path, expected_pngs = make_sampling_input(tmp_path / "input", kind)
    config = dict(
        live_config,
        input_path=str(input_path),
        output_path=str(tmp_path / "evaluation.json"),
        task="Describe the visible colors, then judge whether the final frame is redder than it is green.",
    )
    payload = _run_evaluation(config, strategy=strategy, max_frames=3)
    assert_sampling_evidence(payload, kind, strategy, expected_pngs)


def _assert_self_hosted_judge_claims(saved: dict) -> None:
    from npa.workbench.vlm_eval import _build_prompt, parse_structured_response

    evidence = saved["evidence"]
    response = json.loads(evidence["provider"]["raw_response"])
    verdict = parse_structured_response(response["choices"][0]["message"]["content"])
    assert saved["score"] == round(verdict.score, 4)
    assert saved["passed"] is (saved["score"] >= saved["success_threshold"])
    assert saved["status"] == ("passed" if saved["passed"] else "needs_iteration")
    assert saved["provider_success"] is verdict.provider_success
    matches_gate = (
        None
        if verdict.provider_success is None
        else verdict.provider_success == saved["passed"]
    )
    assert saved["provider_success_matches_score_gate"] is matches_gate
    prompt = _build_prompt(
        **{
            field: saved[field]
            for field in ("task", "rubric", "frame_selection", "frame_count")
        }
    )
    assert (
        evidence["request"]["prompt_sha256"]
        == hashlib.sha256(prompt.encode()).hexdigest()
    )
    assert (
        evidence["request"]["rubric_sha256"]
        == hashlib.sha256(saved["rubric"].encode()).hexdigest()
    )


def _assert_provider_evidence(provider: dict, config: dict) -> None:
    raw_response = provider["raw_response"]
    assert (
        provider["raw_response_sha256"]
        == hashlib.sha256(raw_response.encode()).hexdigest()
    )
    assert provider["status_code"] == 200
    assert provider["returned_model"] == config["expected_served_model"]
    assert provider["finish_reason"] == "stop"
    response = json.loads(raw_response)
    assert response["model"] == config["expected_served_model"]
    assert response["choices"][0]["finish_reason"] == "stop"
    content = response["choices"][0]["message"]["content"].strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE
    )
    expected_parser = "npa_vlm_eval_compatible_json_v2"
    if fence:
        content = fence.group(1)
        expected_parser += "+markdown-fence-v1"
    assert provider["parser_version"] == expected_parser
    assert isinstance(json.loads(content), dict)
