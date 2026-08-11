"""CLI wiring for the npa.workflow runtime tier (`submit --runtime`, `plan-spec --waves`).

The runtime driver itself is mocked: these tests pin the CLI contract — which
options reach ``RuntimeOptions``, what is printed, the exit code, and above all
that the **default** submit path is untouched by the new flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.runtime import RuntimeReport
from npa.orchestration.skypilot.workflow import WorkflowResult

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
FANOUT = SPECS / "token-factory-parallel-fanout.yaml"
GATE_LOOP = SPECS / "token-factory-gate-loop.yaml"
RUNNER = CliRunner()


@pytest.fixture()
def fake_runtime(mocker):
    """Patch the runtime driver and capture how the CLI invoked it."""

    captured: dict[str, object] = {}

    def _run(spec, **kwargs):
        captured["spec"] = spec
        captured.update(kwargs)
        return RuntimeReport(
            workflow=spec.name,
            run_id=str(kwargs.get("run_id") or ""),
            status="succeeded",
            waves=[
                {
                    "key": "001|caption-shards|...",
                    "kind": "parallel",
                    "states": ["caption-shard-a", "caption-shard-b", "caption-shard-c"],
                    "job_id": "42",
                    "status": "succeeded",
                    "max_concurrent_observed": 3,
                },
                {
                    "key": "002|serial|:aggregate:-",
                    "kind": "serial",
                    "states": ["aggregate"],
                    "job_id": "43",
                    "status": "succeeded",
                },
            ],
            decisions=[{"decision": "promote_checkpoint", "uri": "s3://b/gate/decision.json"}],
            run_prefix_uri="s3://b/prefix",
            runtime_state_uri="s3://b/prefix/npa-workflow/runtime.json",
        )

    mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime",
        side_effect=_run,
    )
    return captured


def test_submit_runtime_passes_options_and_emits_json(fake_runtime) -> None:
    config_path = REPO_ROOT / "npa" / "workflows" / "workbench" / "config.yaml"
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-cli-1",
            "--runtime",
            "--config-path",
            str(config_path),
            "--registry",
            "cr.example.invalid/reg",
            "--poll-seconds",
            "7",
            "--max-wait-seconds",
            "123",
            "--retries",
            "2",
            "--max-concurrency",
            "2",
            "--no-cancel-on-timeout",
            "--var",
            "max_images=1",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    options = fake_runtime["options"]
    assert options.poll_seconds == 7
    assert options.max_wait_seconds == 123
    assert options.retries == 2
    assert options.max_concurrency == 2
    assert options.cancel_on_timeout is False
    assert options.resume is False
    assert options.config_path == config_path
    # --var reaches the spec's config, not just the renderer.
    assert fake_runtime["spec"].config["max_images"] == "1"
    assert fake_runtime["render_options"].registry == "cr.example.invalid/reg"

    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["status"] == "succeeded"
    assert payload["wave_count"] == 2
    assert payload["runtime_state_uri"].endswith("/npa-workflow/runtime.json")


def test_submit_runtime_passes_per_tool_image_override(fake_runtime) -> None:
    image = "cr.example.invalid/reg/npa-fiftyone:fixed"
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-tool-image",
            "--runtime",
            "--tool-image",
            f"workbench.fiftyone.curate_augmented={image}",
        ],
    )

    assert result.exit_code == 0, result.output
    options = fake_runtime["render_options"]
    assert options.image_overrides == {
        "workbench.fiftyone.curate_augmented": image,
    }


def test_submit_rejects_malformed_per_tool_image_override() -> None:
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--tool-image",
            "workbench.fiftyone.curate_augmented",
        ],
    )

    assert result.exit_code == 1
    assert "--tool-image must be TOOL_REF=IMAGE" in result.output


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("workbench.fiftyone.curate_augmented", "must be TOOL_REF=IMAGE"),
        (
            "workbench.fiftyone.curate_augmented=registry/fiftyone:test",
            "supported only for npa.workflow/v0.0.1",
        ),
    ],
)
def test_legacy_skypilot_submit_rejects_tool_image_instead_of_ignoring_it(
    tmp_path: Path, override: str, message: str
) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("name: legacy\nresources:\n  cloud: kubernetes\nrun: echo ok\n")
    result = RUNNER.invoke(
        app,
        ["workbench", "workflow", "submit", str(legacy), "--tool-image", override],
    )
    assert result.exit_code == 1
    assert message in result.output


def test_submit_runtime_resume_flag_is_forwarded(fake_runtime) -> None:
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-cli-resume",
            "--runtime",
            "--resume",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_runtime["options"].resume is True


def test_submit_runtime_text_output_lists_waves_and_decisions(fake_runtime) -> None:
    result = RUNNER.invoke(
        app,
        ["workbench", "workflow", "submit", str(FANOUT), "--run-id", "rt-cli-2", "--runtime"],
    )
    assert result.exit_code == 0, result.output
    assert "status: succeeded" in result.output
    assert "waves: 2" in result.output
    assert "[parallel]" in result.output
    assert "decision: promote_checkpoint" in result.output


def test_submit_runtime_failure_exits_non_zero(mocker) -> None:
    mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime",
        side_effect=lambda spec, **kwargs: RuntimeReport(
            workflow=spec.name,
            run_id="rt-cli-fail",
            status="failed",
            error="wave 001 reached terminal status FAILED",
        ),
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-cli-fail",
            "--runtime",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["status"] == "failed"
    assert "terminal status FAILED" in payload["error"]


def test_submit_without_runtime_uses_the_one_shot_path(mocker, monkeypatch) -> None:
    """Backwards compatibility: the default submit path never calls the driver."""

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    runtime_driver = mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime"
    )
    submitted: dict[str, object] = {}

    def fake_submit(path, run_id, **kwargs):
        submitted["content"] = Path(path).read_text(encoding="utf-8")
        return WorkflowResult(status="SUBMITTED", job_id="9", returncode=0)

    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow", side_effect=fake_submit
    )

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "one-shot-1",
            "--image",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    runtime_driver.assert_not_called()
    # The parallel group is flattened into today's serial pipeline.
    assert "execution: serial" in str(submitted["content"])
    assert "caption-shard-c" in str(submitted["content"])


def test_submit_can_preserve_managed_registry_secret(mocker, monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    refresh = mocker.patch(
        "npa.cli.workbench.workflow._refresh_kubernetes_pull_secrets"
    )
    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        return_value=WorkflowResult(status="SUBMITTED", job_id="9", returncode=0),
    )

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "managed-registry-secret",
            "--no-refresh-registry-secret",
        ],
    )

    assert result.exit_code == 0, result.output
    refresh.assert_not_called()


def test_plan_only_wins_over_runtime(mocker, monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    runtime_driver = mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime"
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "plan-only-runtime",
            "--runtime",
            "--plan-only",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    runtime_driver.assert_not_called()
    payload = json.loads(result.output)
    assert payload["status"] == "PLANNED"
    assert "execution: serial" in payload["skypilot_yaml"]


def test_plan_spec_waves_text_and_json() -> None:
    text_result = RUNNER.invoke(
        app,
        ["workbench", "workflow", "plan-spec", str(FANOUT), "--run-id", "w1", "--waves"],
    )
    assert text_result.exit_code == 0, text_result.output
    assert "waves: 2" in text_result.output
    assert "[parallel] caption-shards" in text_result.output
    assert "maxConcurrency=3" in text_result.output

    json_result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(FANOUT),
            "--run-id",
            "w1",
            "--waves",
            "--json",
        ],
    )
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert payload["wave_count"] == 2
    assert payload["parallel_waves"] == 1
    first = payload["waves"][0]
    assert first["kind"] == "parallel"
    assert [step["state"] for step in first["steps"]] == [
        "caption-shard-a",
        "caption-shard-b",
        "caption-shard-c",
    ]


def test_plan_spec_waves_for_a_loop_spec_shows_every_iteration() -> None:
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(GATE_LOOP),
            "--run-id",
            "w2",
            "--waves",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # promote on the first iteration -> caption/score/gate once, then route+publish.
    assert [wave["name"] for wave in payload["waves"]] == [
        "caption-batch",
        "score-batch",
        "quality-gate",
        "route",
        "publish",
    ]
    assert payload["parallel_waves"] == 0
