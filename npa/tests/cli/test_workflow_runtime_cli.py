"""CLI wiring for the npa.workflow runtime tier (`submit --runtime`, `plan-spec --waves`).

The runtime driver itself is mocked: these tests pin the CLI contract — which
options reach ``RuntimeOptions``, what is printed, the exit code, and above all
that the **default** submit path is untouched by the new flags.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.runtime import RuntimeReport
from npa.orchestration.npa_workflow.run_resolution import RunResolution
from npa.orchestration.skypilot.workflow import WorkflowResult

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
FANOUT = SPECS / "token-factory-parallel-fanout.yaml"
GATE_LOOP = SPECS / "token-factory-gate-loop.yaml"
RUNNER = CliRunner()


def test_terminal_ten_wave_runtime_without_active_jobs_stays_succeeded() -> None:
    from npa.cli.workbench.workflow import _manifest_pending_status

    run_id = "paidf-terminal-ten"
    resolution = RunResolution(
        run_id=run_id,
        project="live",
        found=True,
        source="durable_runtime_ledger",
        workflow_name="physical-ai-data-factory",
        run_prefix_uri=f"s3://bucket/physical-ai-data-factory/{run_id}",
        manifest_uri=(
            f"s3://bucket/physical-ai-data-factory/{run_id}/npa-workflow/manifest.json"
        ),
        runtime_state={
            "schema_version": "npa.workflow.runtime.v1",
            "status": "succeeded",
            "waves": [
                {
                    "key": f"wave-{index}",
                    "states": [f"stage-{index}"],
                    "status": "succeeded",
                    "attempt": 1,
                }
                for index in range(10)
            ],
        },
    )
    payload = _manifest_pending_status(
        resolution,
        project="live",
        sky_bin="",
        startup_failure_threshold=3,
    )
    assert payload["status"] == "SUCCEEDED"
    assert payload["status"] != "NOT_SUBMITTED"
    assert payload["verification_status"] == "VERIFIED"
    assert payload["manifest_state"] == "pending"


def test_terminal_status_uses_latest_attempt_without_erasing_history() -> None:
    from npa.cli.workbench.workflow import _latest_runtime_wave_states

    waves = [
        {"key": "001|serial|:prepare:-", "attempt": 1, "status": "succeeded"},
        {"key": "002|serial|:augment:-", "attempt": 1, "status": "failed"},
        {"key": "002|serial|:augment:-", "attempt": 2, "status": "succeeded"},
        {"key": "003|serial|:finalize:-", "attempt": 1, "status": "succeeded"},
    ]

    assert _latest_runtime_wave_states(waves) == {"SUCCEEDED"}
    # The immutable failed attempt is preserved for diagnostics and audit.
    assert waves[1]["status"] == "failed"


def test_terminal_status_keeps_latest_failed_attempt_inconsistent() -> None:
    from npa.cli.workbench.workflow import _latest_runtime_wave_states

    waves = [
        {"key": "001|serial|:augment:-", "attempt": 1, "status": "succeeded"},
        {"key": "001|serial|:augment:-", "attempt": 2, "status": "failed"},
    ]

    assert _latest_runtime_wave_states(waves) == {"FAILED"}


def test_prepare_run_is_fresh_by_default_and_resume_is_explicit(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.orchestration.npa_workflow import first_run_state

    monkeypatch.setattr(first_run_state, "DEFAULT_ROOT", tmp_path / "scoped")
    monkeypatch.setattr(first_run_state, "LEGACY_PATH", tmp_path / "missing-legacy")
    monkeypatch.setattr(
        first_run_state,
        "resolve_project_identity",
        lambda project: (
            "project-stable",
            project or "default",
            "configured_project_id",
        ),
    )

    first = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "prepare-run",
            str(FANOUT),
            "--project",
            "synthetic",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert first_payload["generated_new"] is True
    assert first_payload["resume_explicit"] is False

    second = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "prepare-run",
            str(FANOUT),
            "--project",
            "synthetic",
            "--json",
        ],
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert second_payload["run_id"] != first_payload["run_id"]
    assert second_payload["previous_run"]["run_id"] == first_payload["run_id"]
    assert second_payload["previous_run"]["age_seconds"] is not None

    resumed = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "prepare-run",
            str(FANOUT),
            "--project",
            "synthetic",
            "--resume-run",
            first_payload["run_id"],
            "--json",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    resumed_payload = json.loads(resumed.output)
    assert resumed_payload["run_id"] == first_payload["run_id"]
    assert resumed_payload["generated_new"] is False
    assert resumed_payload["resume_explicit"] is True


@pytest.fixture()
def satisfied_preflight(mocker, monkeypatch):
    """Meet `submit`'s prerequisites so these tests exercise the runtime wiring.

    The runtime path runs the same preflight as the one-shot path (it needs the
    SkyPilot CLI and an npa source for image-less steps); `--var bucket=` in each
    invocation covers the placeholder-bucket check.
    """
    import npa.orchestration.skypilot._bin as skybin
    from npa.clients import storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    mocker.patch.object(skybin, "resolve_sky_bin", lambda _bin: "/usr/bin/sky")
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://rt-bucket/npa-src/npa")
    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(
            True,
            "ok",
            "Writable S3 verified with a cleaned write/delete probe.",
            cleanup_attempted=True,
            cleanup_succeeded=True,
        ),
    )


@pytest.fixture()
def fake_runtime(mocker, satisfied_preflight):
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
            decisions=[
                {"decision": "promote_checkpoint", "uri": "s3://b/gate/decision.json"}
            ],
            run_prefix_uri="s3://b/prefix",
            runtime_state_uri="s3://b/prefix/npa-workflow/runtime.json",
        )

    mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime",
        side_effect=_run,
    )
    return captured


def test_submit_runtime_passes_options_and_emits_json(
    fake_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The runtime automatically forwards project storage credentials even when
    # the caller did not repeat their names with --secret-env.  Its just-in-time
    # resolver must refresh that same expanded set before every wave.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "rotating-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "rotating-secret")
    sky_config = tmp_path / "sky.yaml"
    sky_config.write_text("kubernetes: {}\n", encoding="utf-8")
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(0o755)
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
            "--var",
            "bucket=rt-bucket",
            "--config-path",
            str(sky_config),
            "--sky-bin",
            str(sky_bin),
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
    assert options.retry_absent_in_flight is False
    assert options.max_concurrency == 2
    assert options.cancel_on_timeout is False
    # A run without an explicit --resume-run is always fresh.
    assert options.resume is False
    assert options.config_path == sky_config
    assert options.sky_bin == str(sky_bin)
    assert options.secret_envs == ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    assert options.credential_resolver() == {
        "AWS_ACCESS_KEY_ID": "rotating-access",
        "AWS_SECRET_ACCESS_KEY": "rotating-secret",
    }
    # --var reaches the spec's config, not just the renderer.
    assert fake_runtime["spec"].config["max_images"] == "1"
    assert fake_runtime["render_options"].registry == "cr.example.invalid/reg"

    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["status"] == "succeeded"
    assert payload["wave_count"] == 2
    assert payload["runtime_state_uri"].endswith("/npa-workflow/runtime.json")


def test_retry_absent_in_flight_requires_explicit_resume(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-not-resumed",
            "--runtime",
            "--retry-absent-in-flight",
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    assert "requires an explicit --resume-run ID" in result.output


def test_submit_runtime_passes_explicit_absent_recovery_on_resume(
    fake_runtime,
) -> None:
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--resume-run",
            "rt-explicit-recovery",
            "--runtime",
            "--retry-absent-in-flight",
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_runtime["options"].resume is True
    assert fake_runtime["options"].retry_absent_in_flight is True


def test_submit_runtime_refreshes_pull_secret_before_driver(
    fake_runtime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runtime branch must not bypass private-registry secret refresh."""

    kubeconfig = tmp_path / "pinned-kubeconfig"
    kubeconfig.write_text("current-context: ambient-other-cluster\n", encoding="utf-8")
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._adopt_npa_kubeconfig", lambda _context: True
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._available_kube_contexts", lambda: ["target"]
    )
    monkeypatch.setattr(
        "npa.controller_ownership.verify_controller_owner", lambda *_args: None
    )
    events: list[tuple[str, str, str, str]] = []

    def refresh(
        rendered_path: Path, *, k8s_context: str = "", kubeconfig: str = ""
    ) -> None:
        events.append(("refresh", rendered_path.name, k8s_context, kubeconfig))

    monkeypatch.setattr(
        "npa.cli.workbench.workflow._refresh_kubernetes_pull_secrets", refresh
    )

    original = __import__(
        "npa.orchestration.npa_workflow.runtime", fromlist=["run_workflow_runtime"]
    ).run_workflow_runtime

    def ordered_driver(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        events.append(("driver", str(kwargs.get("run_id") or ""), "", ""))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime", ordered_driver
    )

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-pull-secret-order",
            "--runtime",
            "--infra",
            "k8s/target",
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert events[0] == (
        "refresh",
        "token-factory-parallel-fanout.skypilot.yaml",
        "target",
        str(kubeconfig),
    )
    assert events[1] == ("driver", "rt-pull-secret-order", "", "")
    hook = fake_runtime["options"].pre_submit_hook
    assert hook is not None
    wave = tmp_path / "wave.yaml"
    wave.write_text("name: wave\n", encoding="utf-8")
    hook(wave)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "ambient-after-submit"))
    hook(wave)
    assert events[2:] == [
        ("refresh", "wave.yaml", "target", str(kubeconfig)),
        ("refresh", "wave.yaml", "target", str(kubeconfig)),
    ]


def test_submit_runtime_can_preserve_managed_registry_secret(
    fake_runtime, mocker
) -> None:
    refresh = mocker.patch(
        "npa.cli.workbench.workflow._refresh_kubernetes_pull_secrets"
    )

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-managed-registry-secret",
            "--runtime",
            "--no-refresh-registry-secret",
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_runtime["options"].pre_submit_hook is None
    refresh.assert_not_called()


def test_dynamic_runtime_registry_render_does_not_assume_the_real_gate(
    fake_runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registry setup may choose a render branch; the runtime gate stays real."""

    monkeypatch.setattr(
        "npa.cli.workbench.workflow._preflight_submit_images",
        lambda *_args, **_kwargs: {},
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(GATE_LOOP),
            "--run-id",
            "rt-dynamic-real-gate",
            "--runtime",
            "--image",
            "none",
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_runtime["assume_decision"] == ""


def test_submit_runtime_pinned_no_source_preserves_registry_render_error(
    mocker, monkeypatch: pytest.MonkeyPatch, satisfied_preflight
) -> None:
    """A fail-fast render error must not be masked by cleanup bookkeeping."""
    from npa.orchestration.npa_workflow.skypilot_render import (
        NpaWorkflowRenderError,
    )

    monkeypatch.setattr(
        "npa.cli.workbench.workflow._plan_requires_npa_source",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._preflight_submit_images",
        lambda *_args, **_kwargs: {},
    )
    mocker.patch(
        "npa.orchestration.npa_workflow.submit.prepare_npa_workflow_for_submit",
        side_effect=NpaWorkflowRenderError("expected registry mismatch"),
    )

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-pinned-registry-error",
            "--runtime",
            "--no-stage-src",
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    assert "expected registry mismatch" in result.output
    assert "referenced before assignment" not in result.output
    assert "UnexpectedError" not in result.output


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
            "--var",
            "bucket=rt-bucket",
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
    assert "TOOL_REF=IMAGE" in result.output


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("workbench.fiftyone.curate_augmented", "Use TOOL_REF=IMAGE"),
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
            "--var",
            "bucket=rt-bucket",
            "--resume",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_runtime["options"].resume is True


def test_runtime_uses_configured_secrets_for_local_ledger_without_leaking_env(
    mocker, monkeypatch, satisfied_preflight
) -> None:
    secret = "configured-runtime-secret"
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        lambda **kwargs: type(
            "Context",
            (),
            {
                "endpoint_url": "https://storage.us-central1.nebius.cloud",
                "secret_values": {"AWS_SECRET_ACCESS_KEY": secret},
                "missing": (),
            },
        )(),
    )
    observed: dict[str, str] = {}

    def fake_run(spec, **kwargs):  # noqa: ANN001
        observed["secret"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        observed["endpoint"] = os.environ.get("AWS_ENDPOINT_URL", "")
        return RuntimeReport(
            workflow=spec.name,
            run_id=str(kwargs["run_id"]),
            status="succeeded",
        )

    mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime",
        side_effect=fake_run,
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-configured-creds",
            "--runtime",
            "--var",
            "bucket=rt-bucket",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "secret": secret,
        "endpoint": "https://storage.us-central1.nebius.cloud",
    }
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert secret not in result.output


def test_submit_runtime_text_output_lists_waves_and_decisions(fake_runtime) -> None:
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--run-id",
            "rt-cli-2",
            "--runtime",
            "--var",
            "bucket=rt-bucket",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: succeeded" in result.output
    assert "waves: 2" in result.output
    assert "[parallel]" in result.output
    assert "decision: promote_checkpoint" in result.output


def test_submit_runtime_failure_exits_non_zero(mocker, satisfied_preflight) -> None:
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
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["status"] == "failed"
    assert "terminal status FAILED" in payload["error"]


def test_submit_without_runtime_uses_the_one_shot_path(
    mocker, monkeypatch, satisfied_preflight
) -> None:
    """Backwards compatibility: the default submit path never calls the driver."""

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    runtime_driver = mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime"
    )
    submitted: dict[str, object] = {}
    submit_calls = 0

    def fake_submit(path, run_id, **kwargs):
        nonlocal submit_calls
        submit_calls += 1
        submitted["content"] = Path(path).read_text(encoding="utf-8")
        return WorkflowResult(
            status="SUBMITTED",
            job_id="9",
            returncode=0,
            launch_transaction={
                "state": "adopted" if submit_calls > 1 else "submitted"
            },
        )

    submit_mock = mocker.patch(
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
            "--var",
            "bucket=rt-bucket",
        ],
    )

    assert result.exit_code == 0, result.output
    runtime_driver.assert_not_called()
    # The parallel group is flattened into today's serial pipeline.
    assert "execution: serial" in str(submitted["content"])
    assert "caption-shard-c" in str(submitted["content"])

    resumed = RUNNER.invoke(
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
            "--var",
            "bucket=rt-bucket",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "status: SUBMITTED" in resumed.output
    assert submit_mock.call_count == 2


def test_submit_can_preserve_managed_registry_secret(
    mocker, monkeypatch, satisfied_preflight
) -> None:
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
            "--var",
            "bucket=rt-bucket",
        ],
    )

    assert result.exit_code == 0, result.output
    refresh.assert_not_called()


def test_plan_only_wins_over_runtime(mocker, monkeypatch, satisfied_preflight) -> None:
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
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(FANOUT),
            "--run-id",
            "w1",
            "--waves",
        ],
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


def test_submit_runtime_is_subject_to_the_prerequisite_preflight(mocker) -> None:
    """`--runtime` needs SkyPilot and an npa source just as much as the one-shot path.

    Without them the driver must not be reached: the run would fail later, in the
    controller, with far less to go on.
    """
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
            "rt-cli-preflight",
            "--runtime",
        ],
    )

    assert result.exit_code == 1
    assert "missing prerequisites" in result.output
    runtime_driver.assert_not_called()
