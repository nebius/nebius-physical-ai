from __future__ import annotations

import json
from pathlib import Path
import traceback
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from PIL import Image
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_BENCHMARK_PATH,
    JUDGE_COMPARISON_RESULT_FILENAME,
    LEGACY_RESULT_FILENAME,
    PREFERENCE_COMPARISON_RESULT_FILENAME,
    RESULT_FILENAME,
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
            "--rubric",
            "Require visible completion in both independent reviews.",
            "--frame-selection",
            "sequence",
            "--max-frames",
            "7",
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
    for role in ("primary", "secondary"):
        assert retained[role]["result"]["rubric"] == retained["rubric"]
        assert retained[role]["result"]["provider_success"] is True
        assert retained[role]["result"]["provider_success_matches_score_gate"] is True
        manifest = retained[role]["result"]["evidence"]["request"]["request_manifest"]
        assert manifest["sampling"]["strategy"] == "sequence"
        assert manifest["sampling"]["max_frames"] == 7
        assert manifest["sampling"]["selected_count"] == 1


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
    assert payload["best_config"]["metrics"]["true_negatives"] == 3
    assert payload["schema_version"] == "npa_vlm_eval_benchmark_report_v2"
    assert payload["best_config"]["metrics"]["confusion_matrix"] == {
        "actual_positive": {
            "predicted_positive": 2,
            "predicted_negative": 0,
        },
        "actual_negative": {
            "predicted_positive": 0,
            "predicted_negative": 3,
        },
    }
    assert payload["best_config"]["metrics"]["false_positive_item_ids"] == []
    assert payload["best_config"]["metrics"]["false_negative_item_ids"] == []
    assert payload["written_uri"] == str(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["item_count"] == 5
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


def _preference_cli_images(tmp_path, *, private_names=False):
    first_name = "private-baseline-name.png" if private_names else "first.png"
    second_name = "private-second-name.png" if private_names else "second.png"
    first = tmp_path / first_name
    second = tmp_path / second_name
    Image.new("RGB", (8, 8), "red").save(first)
    Image.new("RGB", (8, 8), "blue").save(second)
    return first, second


def _preference_cli_completion(request, ordinal):
    preference = "B" if ordinal == 1 else "A"
    content = {
        "preference": preference,
        "confidence": "high",
        "observable_support": ["private visible support"],
        "critical_defects": {
            "A": ["private A defect"],
            "B": ["private B defect"],
        },
        "uncertainty": "private uncertainty",
    }
    return {
        "id": f"private-request-{ordinal}",
        "model": request["model"],
        "usage": {"completion_tokens": 12},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(content)},
            }
        ],
    }


def _preference_cli_argv(first, second, output, task, *, json_output):
    argv = [
        "workbench",
        "vlm-eval",
        "compare-preference",
        "--baseline-path",
        str(first),
        "--candidate-path",
        str(second),
        "--output-path",
        str(output),
        "--task",
        task,
        "--rubric",
        "Prefer visible detail.",
    ]
    return [*argv, "--output", "json"] if json_output else argv


def _assert_private_preference_artifacts(output_dir) -> None:
    written = output_dir / PREFERENCE_COMPARISON_RESULT_FILENAME
    retained = json.loads(written.read_text(encoding="utf-8"))
    assert retained["first_order"]["provider"]["provider_request_id"] == (
        "private-request-1"
    )
    assert retained["reversed_order"]["verdict"]["preference"] == "A"
    assert written.stat().st_mode & 0o777 == 0o600
    journal = output_dir / ".vlm_preference_comparison"
    assert {path.name for path in journal.iterdir()} == {
        "state.json",
        "request-01.json",
        "transport-boundary-01.json",
        "response-01.json",
        "request-02.json",
        "transport-boundary-02.json",
        "response-02.json",
        "report-ready.json",
    }
    assert journal.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in journal.iterdir())


def _assert_preference_cli_omits_private_values(result, *values) -> None:
    for value in values:
        assert str(value) not in result.output


def _assert_preference_cli_success(result, requests) -> None:
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "consistent_candidate_preference"
    assert payload["requests_counterbalanced"] is True
    assert payload["artifact_written"] is True
    assert len(requests) == 2


@pytest.mark.parametrize("json_output", [False, True])
def test_preference_cli_rejects_ambiguous_routing_without_transport(
    monkeypatch, tmp_path, json_output
):
    from npa.workbench import vlm_eval

    baseline, candidate = _preference_cli_images(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://private-route.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-private-key")
    post = Mock()
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    result = runner.invoke(
        app,
        _preference_cli_argv(
            baseline,
            candidate,
            tmp_path / "evidence",
            "Compare visible detail.",
            json_output=json_output,
        ),
    )
    assert result.exit_code != 0
    post.assert_not_called()
    _assert_preference_cli_omits_private_values(
        result, baseline, candidate, "private-route.invalid", "synthetic-private-key"
    )


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize(
    ("trigger", "reason", "guidance"),
    [
        ("ambient", "preference_explicit_endpoint_required", "--endpoint-url"),
        ("custom-key", "preference_explicit_endpoint_required", "--endpoint-url"),
        ("missing-key", "preference_named_credential_required", "--api-key-env"),
        ("store", "preference_credential_configuration_invalid", "credential store"),
    ],
)
def test_preference_cli_routing_refusals_are_actionable_and_private(
    monkeypatch, tmp_path, json_output, trigger, reason, guidance
):
    from npa.workbench import vlm_eval

    first, second = _preference_cli_images(tmp_path, private_names=True)
    output = tmp_path / "private-output"
    argv = _preference_cli_argv(
        first, second, output, "Compare visible detail.", json_output=json_output
    )
    _configure_preference_routing_refusal(monkeypatch, argv, trigger, output)
    post = Mock()
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    result = runner.invoke(app, argv)
    _assert_preference_routing_refusal(result, json_output, reason, guidance)
    post.assert_not_called()
    assert not output.exists()
    _assert_preference_cli_omits_private_values(
        result,
        first,
        second,
        output,
        "private-route.invalid",
        "synthetic-private-key",
        "SYNTHETIC_PRIVATE_KEY_NAME",
    )


def _configure_preference_routing_refusal(monkeypatch, argv, trigger, output):
    from npa.clients import token_factory

    for name in ("VLM_EVAL_API_KEY", "NEBIUS_TOKEN_FACTORY_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    if trigger == "ambient":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://private-route.invalid/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-private-key")
    elif trigger == "custom-key":
        argv += ["--api-key-env", "SYNTHETIC_PRIVATE_KEY_NAME"]
    elif trigger == "store":
        monkeypatch.setattr(
            token_factory,
            "resolve_config",
            Mock(side_effect=ValueError(f"synthetic-private-key at {output}")),
        )


def _assert_preference_routing_refusal(result, json_output, reason, guidance):
    assert result.exit_code == 1
    assert reason in result.output
    assert guidance in result.output
    assert "inspect private evidence" not in result.output
    assert "journal" not in result.output
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["result"] == "error"
        assert payload["reason_code"] == reason
        assert guidance in payload["message"]


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize(
    "prefix",
    [
        "preference_explicit_endpoint_required",
        "preference_named_credential_required",
        "preference_credential_configuration_invalid",
        "unrecognized_preference_failure",
    ],
)
def test_preference_cli_never_echoes_non_allowlisted_error_details(
    monkeypatch, tmp_path, json_output, prefix
):
    from npa.cli.workbench import vlm_eval as cli_vlm

    private_detail = "synthetic-private-key https://private-route.invalid /private/path"
    monkeypatch.setattr(
        cli_vlm,
        "_execute_preference_comparison",
        Mock(side_effect=cli_vlm.VlmEvalError(f"{prefix}: {private_detail}")),
    )
    first, second = _preference_cli_images(tmp_path)
    result = runner.invoke(
        app,
        _preference_cli_argv(
            first,
            second,
            tmp_path / "output",
            "Visible detail.",
            json_output=json_output,
        ),
    )
    assert result.exit_code == 1
    assert "Blinded preference comparison failed" in result.output
    _assert_preference_cli_omits_private_values(result, prefix, private_detail)


def test_workbench_vlm_eval_compare_preference_writes_private_report(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    baseline, candidate = _preference_cli_images(tmp_path)
    requests = []

    def post(**kwargs):
        request = kwargs["request"]
        requests.append(request)
        return _preference_cli_completion(request, len(requests))

    monkeypatch.setenv("VLM_EVAL_API_KEY", "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    output_dir = tmp_path / "preference"
    argv = _preference_cli_argv(
        baseline,
        candidate,
        output_dir,
        "Compare matched scene views.",
        json_output=True,
    )
    result = runner.invoke(app, argv)

    _assert_preference_cli_success(result, requests)
    _assert_preference_cli_omits_private_values(
        result,
        baseline,
        candidate,
        output_dir,
        "private visible support",
        "private uncertainty",
        "private-request-1",
        "raw_response",
    )
    _assert_private_preference_artifacts(output_dir)


def test_workbench_vlm_eval_compare_preference_sanitizes_failure(
    monkeypatch, tmp_path
) -> None:
    from npa.workbench import vlm_eval

    baseline, candidate = _preference_cli_images(tmp_path, private_names=True)
    called = False

    def post(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    secret_task = "Prefer the candidate over the baseline for operator-task-17."
    argv = _preference_cli_argv(
        baseline,
        candidate,
        tmp_path / "private-output",
        secret_task,
        json_output=False,
    )
    result = runner.invoke(app, argv)

    assert result.exit_code == 1
    assert "Blinded preference comparison failed" in result.output
    assert str(baseline) not in result.output
    assert str(candidate) not in result.output
    assert secret_task not in result.output
    assert called is False


def _failing_preference_storage(monkeypatch, stage, error):
    from npa.clients.storage import StorageClient

    client = Mock()
    client.read_bytes_with_etag.return_value = None
    factory = Mock(return_value=client)
    boundary = {
        "configuration": factory,
        "preflight": client.read_bytes_with_etag,
        "input": client.download_path,
        "journal": client.put_bytes_conditional,
    }.get(stage)
    if boundary is not None:
        boundary.side_effect = error
    else:

        def write(body, uri, **kwargs):
            if uri.endswith("/vlm_preference_comparison.json"):
                raise error

        client.put_bytes_conditional.side_effect = write
    monkeypatch.setattr(StorageClient, "from_environment", factory)


@pytest.mark.parametrize(
    "stage", ["configuration", "preflight", "input", "journal", "final"]
)
@pytest.mark.parametrize("error_type", ["provider", "connection", "configuration"])
@pytest.mark.parametrize("json_output", [False, True])
def test_preference_cli_redacts_storage_failures(
    monkeypatch, tmp_path, stage, error_type, json_output
):
    from npa.workbench import vlm_eval

    private_detail = "synthetic-private-storage-detail"
    errors = {
        "provider": ClientError(
            {"Error": {"Code": "AccessDenied", "Message": private_detail}}, "GetObject"
        ),
        "connection": EndpointConnectionError(endpoint_url=private_detail),
        "configuration": ValueError(private_detail),
    }
    _failing_preference_storage(monkeypatch, stage, errors[error_type])
    baseline, candidate = _preference_cli_images(tmp_path)
    if stage == "input":
        baseline = "s3://private-role/source.png"
    post = Mock(
        side_effect=lambda **kwargs: _preference_cli_completion(kwargs["request"], 1)
    )
    monkeypatch.setenv("VLM_EVAL_API_KEY", "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    argv = _preference_cli_argv(
        baseline,
        candidate,
        "s3://private-role/evidence",
        "Compare matched views.",
        json_output=json_output,
    )

    result = runner.invoke(app, argv)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Blinded preference comparison failed" in result.output
    assert private_detail not in result.output
    assert "private-role" not in result.output
    assert post.call_count == (2 if stage == "final" else 0)


@pytest.mark.parametrize("boundary", ["input", "output"])
def test_preference_cli_symlink_loop_paths_are_sanitized(
    tmp_path, monkeypatch, boundary
):
    from npa.workbench import vlm_eval

    first, second = _preference_cli_images(tmp_path)
    loop = tmp_path / "private-loop"
    loop.symlink_to(loop.name)
    post = Mock(side_effect=AssertionError("must not call provider"))
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    output = loop if boundary == "output" else tmp_path / "output"
    baseline = loop if boundary == "input" else first
    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "compare-preference",
            "--baseline-path",
            str(baseline),
            "--candidate-path",
            str(second),
            "--output-path",
            str(output),
            "--task",
            "Compare visible shapes.",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output
    assert "private-loop" not in result.output
    post.assert_not_called()


def test_vlm_eval_sdk_exports_blinded_preference_surface() -> None:
    from npa.sdk.workbench import vlm_eval as sdk_vlm_eval
    from npa.workbench import vlm_eval as core_vlm_eval
    from npa.workbench.vlm_eval import (
        VlmPreferenceComparisonRequest,
        compare_vlm_preference,
    )

    assert sdk_vlm_eval.compare_preference is compare_vlm_preference
    assert sdk_vlm_eval.VlmPreferenceComparisonRequest is VlmPreferenceComparisonRequest
    assert "VlmPreferenceComparisonRequest" in core_vlm_eval.__all__


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


def test_review_visual_default_text_under_json_contract(monkeypatch, tmp_path):
    import npa.cli.workbench.vlm_eval as cli_vlm_eval

    monkeypatch.setattr(
        cli_vlm_eval,
        "run_visual_review",
        lambda _request: SimpleNamespace(
            schema_version="npa_vlm_visual_review_v1",
            status="completed",
            escalation_required=False,
            attempt_count=2,
            model="hosted/vision-model",
        ),
    )
    arguments = _visual_review_cli_args(tmp_path)
    assert arguments[-2:] == ["--output-format", "json"]
    result = runner.invoke(app, arguments[:-2])
    assert result.exit_code == 0
    assert "schema_version: npa_vlm_visual_review_v1" in result.output
    assert "status: completed" in result.output
    assert "attempt_count: 2" in result.output
    assert "private-current" not in result.output


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
