from __future__ import annotations

import json
from pathlib import Path
import traceback
from types import SimpleNamespace

from PIL import Image

from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_BENCHMARK_PATH,
    VlmEvalResult,
    VlmVisualReviewRequest,
    VISUAL_REVIEW_RESULT_FILENAME,
)


runner = CliRunner()


def test_workbench_vlm_eval_command_help() -> None:
    result = runner.invoke(app, ["workbench", "vlm-eval", "--help"])

    assert result.exit_code == 0
    assert "VLM evaluation" in result.output


def test_workbench_vlm_eval_run_writes_local_json(tmp_path) -> None:
    output_dir = tmp_path / "eval"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "s3://bucket/cosmos/out/",
            "--output-path",
            str(output_dir),
            "--backend",
            "stub",
            "--score",
            "0.9",
            "--success-threshold",
            "0.8",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["backend"] == "stub"
    assert payload["passed"] is True
    written = output_dir / "vlm_eval_stub.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["score"] == 0.9


def test_workbench_vlm_eval_dry_run_does_not_write(tmp_path) -> None:
    output_dir = tmp_path / "eval"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "s3://bucket/cosmos/out/",
            "--output-path",
            str(output_dir),
            "--backend",
            "stub",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert not output_dir.exists()


def test_workbench_vlm_eval_respects_env_dry_run(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "eval"
    monkeypatch.setenv("NPA_DRY_RUN", "1")

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "s3://bucket/cosmos/out/",
            "--output-path",
            str(output_dir),
            "--backend",
            "stub",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["dry_run"] is True
    assert not output_dir.exists()


def test_workbench_vlm_eval_run_maps_backend_flags(mocker, tmp_path) -> None:
    output_dir = tmp_path / "eval"
    mock_eval = mocker.patch(
        "npa.cli.workbench.vlm_eval.evaluate_vlm",
        return_value=VlmEvalResult(
            status="passed",
            backend="api",
            input_path="rollouts",
            output_path=str(output_dir),
            result_uri=str(output_dir / "vlm_eval_stub.json"),
            task="place cube",
            model="open-vlm",
            score=0.82,
            success_threshold=0.7,
            passed=True,
            generated_at="2026-01-01T00:00:00+00:00",
            frame_selection="sequence",
            frame_count=8,
            rationale="Object is placed correctly.",
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "rollouts",
            "--output-path",
            str(output_dir),
            "--task",
            "place cube",
            "--backend",
            "api",
            "--model",
            "open-vlm",
            "--endpoint-url",
            "https://vlm.example/v1",
            "--api-key-env",
            "VLM_TOKEN",
            "--frame-selection",
            "sequence",
            "--max-frames",
            "8",
            "--rubric",
            "strict",
            "--success-threshold",
            "0.7",
            "--timeout-s",
            "45",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["backend"] == "api"
    assert payload["frame_selection"] == "sequence"
    kwargs = mock_eval.call_args.kwargs
    assert kwargs["backend"] == "api"
    assert kwargs["model"] == "open-vlm"
    assert kwargs["endpoint_url"] == "https://vlm.example/v1"
    assert kwargs["api_key_env"] == "VLM_TOKEN"
    assert kwargs["frame_selection"] == "sequence"
    assert kwargs["max_frames"] == 8
    assert kwargs["rubric"] == "strict"
    assert kwargs["success_threshold"] == 0.7
    assert kwargs["timeout_s"] == 45


def test_workbench_vlm_eval_workflow_path() -> None:
    result = runner.invoke(
        app, ["workbench", "vlm-eval", "workflow", "--output", "json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    # The advertised path is the npa.workflow spec, not the SkyPilot template.
    assert payload["workflow"] == "workflows/testing/vlm-eval-single.yaml"


def test_workbench_vlm_eval_benchmark_writes_report(tmp_path) -> None:
    output_path = tmp_path / "benchmark-report.json"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "benchmark",
            "--dataset",
            str(DEFAULT_SAMPLE_BENCHMARK_PATH),
            "--output",
            str(output_path),
            "--backend",
            "stub",
            "--thresholds",
            "0.5,0.8,0.9",
            "--rubrics",
            "default,strict",
            "--models",
            DEFAULT_MODEL,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["best_config"]["config"]["success_threshold"] == 0.8
    assert payload["best_config"]["metrics"]["accuracy"] == 1.0
    assert payload["best_config"]["metrics"]["true_positives"] == 2
    assert payload["best_config"]["metrics"]["true_negatives"] == 2
    assert payload["written_uri"] == str(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8"))["item_count"] == 4


def test_vlm_eval_sdk_benchmark_returns_report() -> None:
    from npa.sdk.workbench import vlm_eval as sdk_vlm_eval

    report = sdk_vlm_eval.benchmark(
        dataset=str(DEFAULT_SAMPLE_BENCHMARK_PATH),
        backend="stub",
        thresholds=[0.5, 0.8, 0.9],
        rubrics=["default"],
        models=[DEFAULT_MODEL],
    )

    assert report.best_config.config.success_threshold == 0.8
    assert report.best_config.metrics.accuracy == 1.0


def test_vlm_eval_sdk_wrapper_accepts_string_flags(capsys, tmp_path) -> None:
    from npa.sdk.workbench import vlm_eval as sdk_vlm_eval

    output_dir = tmp_path / "sdk-eval"

    sdk_vlm_eval.run(
        input_path="rollouts",
        output_path=str(output_dir),
        backend="stub",
        frame_selection="final",
        score=0.72,
        output="json",
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "stub"
    assert payload["frame_selection"] == "final"
    assert payload["score"] == 0.72
    assert (output_dir / "vlm_eval_stub.json").exists()


def _visual_review_cli_args(tmp_path: Path) -> list[str]:
    return [
        "workbench",
        "vlm-eval",
        "review-visual",
        "--input-path",
        str(tmp_path / "private-current"),
        "--output-path",
        str(tmp_path / "private-output"),
        "--model",
        "hosted/vision-model",
        "--task",
        "Inspect private visible evidence.",
        "--baseline-path",
        str(tmp_path / "private-reference"),
        "--frame-selection",
        "sequence",
        "--max-frames",
        "7",
        "--rubric",
        "Private review instructions.",
        "--rubric-path",
        str(tmp_path / "private-rubric.txt"),
        "--objective-evidence-path",
        str(tmp_path / "private-objective.json"),
        "--matched-view-map-path",
        str(tmp_path / "private-views.json"),
        "--endpoint-url",
        "https://private-endpoint.invalid/v1",
        "--api-key-env",
        "PRIVATE_REVIEW_TOKEN",
        "--timeout-s",
        "45",
        "--output-format",
        "json",
    ]


def test_workbench_vlm_eval_review_visual_maps_exact_bounded_request(
    monkeypatch, tmp_path: Path
) -> None:
    import npa.cli.workbench.vlm_eval as cli_vlm_eval

    requests = []

    def review(request):
        requests.append(request)
        return SimpleNamespace(
            schema_version="npa_vlm_visual_review_v1",
            status="completed",
            escalation_required=False,
            attempt_count=2,
            model="hosted/vision-model",
        )

    monkeypatch.setenv("PRIVATE_REVIEW_TOKEN", "private-api-secret")
    monkeypatch.setattr(cli_vlm_eval, "run_visual_review", review)
    result = runner.invoke(app, _visual_review_cli_args(tmp_path))

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "schema_version": "npa_vlm_visual_review_v1",
        "status": "completed",
        "escalation_required": False,
        "attempt_count": 2,
        "model": "hosted/vision-model",
    }
    assert requests == [_expected_visual_review_request(tmp_path)]
    for private_value in (
        "private-current",
        "private-output",
        "private-reference",
        "private-rubric",
        "private-objective",
        "private-views",
        "private-endpoint",
        "private-api-secret",
        "Inspect private",
        "Private review",
    ):
        assert private_value not in result.output


def _expected_visual_review_request(tmp_path: Path) -> VlmVisualReviewRequest:
    return VlmVisualReviewRequest(
        input_path=str(tmp_path / "private-current"),
        output_path=str(tmp_path / "private-output"),
        model="hosted/vision-model",
        task="Inspect private visible evidence.",
        baseline_path=str(tmp_path / "private-reference"),
        frame_selection="sequence",
        max_frames=7,
        endpoint_url="https://private-endpoint.invalid/v1",
        api_key_env="PRIVATE_REVIEW_TOKEN",
        rubric="Private review instructions.",
        rubric_path=str(tmp_path / "private-rubric.txt"),
        objective_evidence_path=str(tmp_path / "private-objective.json"),
        matched_view_map_path=str(tmp_path / "private-views.json"),
        timeout_s=45.0,
    )


def test_workbench_vlm_eval_review_visual_sanitizes_generic_failure(
    monkeypatch, tmp_path: Path
) -> None:
    import npa.cli.workbench.vlm_eval as cli_vlm_eval

    private_detail = str(tmp_path / "private-provider-response")

    def fail(_request):
        raise RuntimeError(f"provider failed at {private_detail}")

    monkeypatch.setattr(cli_vlm_eval, "run_visual_review", fail)
    result = runner.invoke(app, _visual_review_cli_args(tmp_path))

    assert result.exit_code == 1
    assert "Visual review failed; inspect private evidence." in result.output
    assert private_detail not in result.output
    assert "provider failed" not in result.output
    assert "private-api-secret" not in result.output


def test_workbench_vlm_eval_review_visual_rejects_alternate_name_pretransport(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.workbench import vlm_eval

    called = False

    def post(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(vlm_eval, "_post_comparison_request", post)
    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "review-visual",
            "--input-path",
            str(tmp_path / "private-missing-input"),
            "--output-path",
            str(tmp_path / "private-other-name.json"),
            "--model",
            "hosted/vision-model",
            "--task",
            "Inspect visible evidence.",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    assert called is False
    assert "private-other-name.json" not in result.output
    assert "private-missing-input" not in result.output


def test_vlm_eval_sdk_review_visual_delegates_matching_request(
    mocker, tmp_path: Path
) -> None:
    from npa.sdk.workbench import vlm_eval as sdk_vlm_eval

    expected = _expected_visual_review_request(tmp_path)
    finalized_report = object()
    backend = mocker.patch.object(
        sdk_vlm_eval, "_review_visual_backend", return_value=finalized_report
    )

    result = sdk_vlm_eval.review_visual(
        input_path=expected.input_path,
        output_path=expected.output_path,
        model=expected.model,
        task=expected.task,
        baseline_path=expected.baseline_path,
        frame_selection=expected.frame_selection,
        max_frames=expected.max_frames,
        endpoint_url=expected.endpoint_url,
        api_key_env=expected.api_key_env,
        rubric=expected.rubric,
        rubric_path=expected.rubric_path,
        objective_evidence_path=expected.objective_evidence_path,
        matched_view_map_path=expected.matched_view_map_path,
        timeout_s=expected.timeout_s,
    )

    assert result is finalized_report
    backend.assert_called_once_with(expected)


def test_vlm_eval_sdk_review_visual_sanitizes_traceback(mocker, tmp_path: Path) -> None:
    from npa.sdk.workbench import vlm_eval as sdk_vlm_eval

    sentinel = str(tmp_path / "private-endpoint-provider-sentinel")
    mocker.patch.object(
        sdk_vlm_eval,
        "_review_visual_backend",
        side_effect=RuntimeError(sentinel),
    )

    try:
        sdk_vlm_eval.review_visual(
            input_path=sentinel,
            output_path=sentinel,
            model="private/model",
            task="Private task.",
        )
    except sdk_vlm_eval.VlmVisualReviewError as exc:
        rendered = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        assert str(exc) == "Visual review failed; inspect private evidence."
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True
        assert sentinel not in rendered
    else:
        raise AssertionError("SDK failure was not bounded")


def _visual_review_single_payload() -> dict:
    assertion = {"text": "Visible scene evidence.", "frame_ids": ["A0001"]}
    return {
        "arm": {
            "task_evidence": {
                "visible_status": "complete",
                "observations": [assertion],
                "hidden_state_limits": ["Pixels do not establish hidden state."],
            },
            "artifact_fidelity": {
                "status": "no_visible_issue",
                "issues": [],
                "uncertainty": "Unseen views remain unknown.",
            },
            "reviewability": {
                "status": "reviewable",
                "strengths": [assertion],
                "limitations": [],
            },
            "impressiveness": {
                "status": "moderate",
                "visible_basis": [assertion],
                "cosmetic_only": False,
            },
            "physical_ai_usefulness": {
                "status": "unsupported",
                "visible_basis": [assertion],
                "downstream_operation": None,
                "required_properties": [],
                "hypothesis": None,
                "measured_consumer_test_needed": None,
            },
        }
    }


def _sdk_visual_review_post(*, request, response_sink, **_kwargs):
    from npa.workbench import vlm_eval

    completion = {
        "model": request["model"],
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(_visual_review_single_payload())},
            }
        ],
    }
    raw = vlm_eval._canonical_json(completion)
    response = vlm_eval._VlmBackendResponse(
        completion, raw, 200, "private-request-id", 0.1
    )
    response_sink(response)
    return response, None


def test_vlm_eval_sdk_review_visual_returns_finalized_report(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.sdk.workbench import vlm_eval as sdk_vlm_eval
    from npa.workbench import vlm_eval

    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "green").save(frame)

    monkeypatch.setattr(
        vlm_eval.visual_review,
        "_resolve_provider_key",
        lambda *_args, **_kwargs: "secret",
    )
    monkeypatch.setattr(vlm_eval, "_post_comparison_request", _sdk_visual_review_post)
    output = tmp_path / "sdk-finalized"
    report = sdk_vlm_eval.review_visual(
        input_path=str(frame),
        output_path=str(output),
        model="hosted/vision-model",
        task="Assess visible placement evidence.",
    )

    written = output / VISUAL_REVIEW_RESULT_FILENAME
    assert report.result_uri == str(written)
    assert written.exists()
    assert written.stat().st_mode & 0o777 == 0o600
    assert json.loads(written.read_text())["status"] == "completed"
