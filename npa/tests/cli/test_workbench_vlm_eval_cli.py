from __future__ import annotations

import json

from PIL import Image
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_BENCHMARK_PATH,
    JUDGE_COMPARISON_RESULT_FILENAME,
    LEGACY_RESULT_FILENAME,
    RESULT_FILENAME,
    VlmEvalResult,
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
    written = output_dir / RESULT_FILENAME
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["score"] == 0.9
    assert not (output_dir / LEGACY_RESULT_FILENAME).exists()


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
            result_uri=str(output_dir / RESULT_FILENAME),
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


def test_workbench_vlm_eval_compare_judges_writes_distinct_report(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "green").save(frame)
    requests = []

    def post(**kwargs):
        request = kwargs["request"]
        requests.append(request)
        return {
            "id": f"request-{len(requests)}",
            "model": request["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"success":true,"score":0.9,'
                            '"rationale":"visible evidence"}'
                        )
                    },
                }
            ],
        }

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    output_dir = tmp_path / "comparison"
    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "compare-judges",
            "--input-path",
            str(frame),
            "--output-path",
            str(output_dir),
            "--primary-model",
            "MiniMaxAI/MiniMax-M3",
            "--secondary-model",
            "openbmb/MiniCPM-V-4_5",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "judges_agree_passed"
    assert payload["deployment_status"] == "audit_only"
    assert payload["requests_differ_only_by_model"] is True
    assert len(requests) == 2
    assert "raw_response" not in result.output
    assert "visible evidence" not in result.output
    written = output_dir / JUDGE_COMPARISON_RESULT_FILENAME
    assert written.exists()
    assert not (output_dir / RESULT_FILENAME).exists()
    retained = json.loads(written.read_text(encoding="utf-8"))
    assert retained["primary"]["result"]["result_uri"] == str(written)
    assert retained["secondary"]["result"]["result_uri"] == str(written)
    assert retained["primary"]["result"]["rationale"] == "visible evidence"


def test_workbench_vlm_eval_compare_judges_rejects_same_model(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "green").save(frame)
    called = False

    def post(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "compare-judges",
            "--input-path",
            str(frame),
            "--output-path",
            str(tmp_path / "comparison"),
            "--primary-model",
            "same/model",
            "--secondary-model",
            "same/model",
        ],
    )

    assert result.exit_code == 1
    assert "two distinct model IDs" in result.output
    assert called is False


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
    assert payload["schema_version"] == "npa_vlm_eval_benchmark_report_v2"
    assert payload["best_config"]["metrics"]["confusion_matrix"] == {
        "actual_positive": {
            "predicted_positive": 2,
            "predicted_negative": 0,
        },
        "actual_negative": {
            "predicted_positive": 0,
            "predicted_negative": 2,
        },
    }
    assert payload["best_config"]["metrics"]["false_positive_item_ids"] == []
    assert payload["best_config"]["metrics"]["false_negative_item_ids"] == []
    assert payload["written_uri"] == str(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["item_count"] == 4
    assert written["schema_version"] == "npa_vlm_eval_benchmark_report_v2"
    assert (
        written["ranked_configs"][0]["metrics"]["confusion_matrix"]
        == payload["best_config"]["metrics"]["confusion_matrix"]
    )
    for ranked in written["ranked_configs"]:
        metrics = ranked["metrics"]
        matrix = metrics["confusion_matrix"]
        assert (
            matrix["actual_positive"]["predicted_positive"] == metrics["true_positives"]
        )
        assert (
            matrix["actual_positive"]["predicted_negative"]
            == metrics["false_negatives"]
        )
        assert (
            matrix["actual_negative"]["predicted_positive"]
            == metrics["false_positives"]
        )
        assert (
            matrix["actual_negative"]["predicted_negative"] == metrics["true_negatives"]
        )
        negative_count = metrics["false_positives"] + metrics["true_negatives"]
        positive_count = metrics["false_negatives"] + metrics["true_positives"]
        assert metrics["false_positive_rate"] == round(
            metrics["false_positives"] / negative_count, 4
        )
        assert metrics["false_negative_rate"] == round(
            metrics["false_negatives"] / positive_count, 4
        )
        assert len(metrics["false_positive_item_ids"]) == metrics["false_positives"]
        assert len(metrics["false_negative_item_ids"]) == metrics["false_negatives"]


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


def test_vlm_eval_sdk_exports_direct_paired_judge_surface() -> None:
    from npa.sdk.workbench import vlm_eval as sdk_vlm_eval
    from npa.workbench import vlm_eval as core_vlm_eval
    from npa.workbench.vlm_eval import (
        VlmJudgeComparisonRequest,
        compare_vlm_judges,
    )

    assert sdk_vlm_eval.compare_judges is compare_vlm_judges
    assert sdk_vlm_eval.VlmJudgeComparisonRequest is VlmJudgeComparisonRequest
    assert "VlmJudgeComparisonRequest" in core_vlm_eval.__all__


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
    assert (output_dir / RESULT_FILENAME).exists()
    assert not (output_dir / LEGACY_RESULT_FILENAME).exists()
