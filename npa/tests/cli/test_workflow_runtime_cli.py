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
from npa.cli.workbench.workflow import (
    _execution_target_preflight as REAL_EXECUTION_TARGET_PREFLIGHT,
)
from npa.orchestration.npa_workflow.runtime import RuntimeReport
from npa.orchestration.npa_workflow.run_resolution import RunResolution
from npa.orchestration.npa_workflow.run_state import RunManifest
from npa.orchestration.skypilot.workflow import SkyPilotSubmitError, WorkflowResult
from npa.orchestration.skypilot.workflow_state import WorkflowS3Config

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "workflows" / "testing"
FANOUT = SPECS / "token-factory-parallel-fanout.yaml"
GATE_LOOP = SPECS / "token-factory-gate-loop.yaml"
PAIDF_COSMOS3 = REPO_ROOT / "workflows" / "main" / "paidf-cosmos3.yaml"
RUNNER = CliRunner()


def _selected_storage_credentials():
    from npa.orchestration.npa_workflow.submit_credentials import (
        SubmitCredentialContext,
    )

    return SubmitCredentialContext(
        endpoint_url="https://storage.us-central1.nebius.cloud",
        access_key_id="selected-access",
        secret_access_key="selected-secret",
    )


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


def test_pending_status_projects_runtime_resource_snapshot_and_preview_error() -> None:
    from npa.cli.workbench.workflow import _manifest_pending_status

    resolution = RunResolution(
        run_id="resource-snapshot",
        project="live",
        found=True,
        source="durable_runtime_ledger",
        workflow_name="training",
        run_prefix_uri="s3://bucket/training/resource-snapshot",
        manifest_uri="s3://bucket/training/resource-snapshot/manifest.json",
        receipt={
            "workflow": {
                "api_version": "npa.workflow/v0.0.1",
                "steps": [],
                "plan_preview": {
                    "status": "failed",
                    "error": "ValueError: decision unavailable",
                },
            }
        },
        runtime_state={
            "schema_version": "npa.workflow.runtime.v1",
            "status": "running",
            "waves": [
                {
                    "key": "wave-train",
                    "states": ["train"],
                    "status": "running",
                    "attempt": 1,
                    "resource_profiles": {
                        "train": {"accelerators": "B200:1", "cpus": 16}
                    },
                }
            ],
        },
    )

    payload = _manifest_pending_status(
        resolution,
        project="live",
        sky_bin="",
        startup_failure_threshold=3,
        cached=True,
    )

    stage = next(iter(payload["stages"].values()))
    assert stage["requested_accelerators"] == "B200:1"
    assert stage["resources_profile"] == {"accelerators": "B200:1", "cpus": 16}
    assert any("decision unavailable" in item for item in payload["diagnostics"])


def test_available_manifest_projects_runtime_resource_snapshot(mocker) -> None:
    from npa.cli.workbench.workflow import _durable_workflow_status

    profile = {"accelerators": "GPU:1", "cpus": "16+", "memory": "128+"}
    manifest = RunManifest(
        "training", "resource-snapshot", "npa.workflow/v0.0.1", status="running"
    )
    resolution = RunResolution(
        run_id="resource-snapshot",
        project="test",
        found=True,
        source="durable_runtime_ledger",
        manifest=manifest.to_dict(),
        runtime_state={
            "schema_version": "npa.workflow.runtime.v1",
            "status": "running",
            "waves": [
                {
                    "key": "001|serial|:qualify:-",
                    "states": ["qualify"],
                    "status": "running",
                    "attempt": 1,
                    "resource_profiles": {"qualify": profile},
                }
            ],
        },
        state=WorkflowS3Config(
            bucket="bucket",
            prefix="resource-snapshot/npa-workflow",
            endpoint_url="https://storage.example.test",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        ),
    )
    mocker.patch(
        "npa.orchestration.npa_workflow.run_resolution.resolve_run",
        return_value=resolution,
    )
    latest = mocker.patch(
        "npa.orchestration.npa_workflow.supervisor.SupervisorLedger.latest",
        return_value=None,
    )

    payload = _durable_workflow_status("resource-snapshot", cached=True)

    assert payload["stages"]["qualify"]["requested_accelerators"] == "GPU:1"
    assert payload["stages"]["qualify"]["resources_profile"] == profile
    latest.assert_called_once_with(run_id="resource-snapshot")


def test_runtime_resource_snapshot_preserves_manifest_profile() -> None:
    from npa.cli.workbench.workflow import _merge_runtime_resource_profiles

    existing = {"accelerators": "GPU:1", "memory": "64+", "source": "manifest"}
    steps = [{"state": "qualify", "resources_profile": existing.copy()}]
    waves = [
        {
            "resource_profiles": {
                "qualify": {
                    "accelerators": "OTHER:8",
                    "memory": "512+",
                    "source": "runtime",
                }
            }
        }
    ]

    _merge_runtime_resource_profiles(steps, waves)

    assert steps[0]["resources_profile"] == existing


def test_runtime_submission_receipt_preserves_redacted_preview_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.workbench.workflow import _runtime_submission_receipt
    from npa.orchestration.npa_workflow import load_spec
    from npa.orchestration.npa_workflow import runtime

    def fail_preview(*_args, **_kwargs):
        raise ValueError("AWS_SECRET_ACCESS_KEY=do-not-persist")

    monkeypatch.setattr(runtime, "plan_preview", fail_preview)
    receipt = _runtime_submission_receipt(load_spec(FANOUT), "preview-run", "")

    assert receipt["steps"] == []
    assert receipt["plan_preview"]["status"] == "failed"
    assert "do-not-persist" not in receipt["plan_preview"]["error"]
    assert "<redacted>" in receipt["plan_preview"]["error"]


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

    mocker.patch(
        "npa.cli.workbench.workflow._execution_target_preflight",
        return_value=(None, {}),
    )

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
        from npa.orchestration.npa_workflow.submission_state import (
            load_submission_state,
            submission_proves_never_launched,
        )

        receipt = load_submission_state(kwargs["options"].project, kwargs["run_id"])
        assert receipt["launch"] == {"status": "launching", "kind": "runtime"}
        assert not submission_proves_never_launched(
            receipt,
            project=kwargs["options"].project,
            run_id=kwargs["run_id"],
        )
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


@pytest.mark.parametrize("selected", ["", "review"])
def test_runtime_keeps_configured_project_selection(
    fake_runtime,
    monkeypatch,
    tmp_path: Path,
    selected: str,
) -> None:
    from npa.clients import config

    path = tmp_path / "project-config.yaml"
    path.write_text(
        "default_project: research\nprojects:\n  research: {}\n  review: {}\n"
    )
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    arguments = [
        "workbench",
        "workflow",
        "submit",
        str(FANOUT),
        "--runtime",
        "--run-id",
        "selected-project-run",
        "--var",
        "bucket=rt-bucket",
        "--output-format",
        "json",
    ]
    if selected:
        arguments.extend(["--project", selected])

    result = RUNNER.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert fake_runtime["options"].project == (selected or "research")


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


@pytest.fixture()
def gpu_then_cpu_spec(tmp_path: Path) -> Path:
    spec = tmp_path / "gpu-then-cpu.yaml"
    spec.write_text("""apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: gpu-then-cpu}
config: {bucket: rt-bucket, prefix: pipeline}
resources:
  gpu: {cloud: kubernetes, accelerators: B200:1, cpus: 16, memory: 128Gi}
  cpu: {cloud: kubernetes, cpus: 4, memory: 16Gi}
initial: generate
states:
  generate: {resources: gpu, run: {shell: 'true'}, next: publish}
  publish: {resources: cpu, run: {shell: 'true'}, terminal: true}
""")
    return spec


@pytest.mark.parametrize("resume", [False, True])
def test_runtime_defers_free_capacity_to_the_actual_wave(
    fake_runtime,
    gpu_then_cpu_spec: Path,
    tmp_path: Path,
    mocker,
    resume: bool,
) -> None:
    capacity = mocker.patch(
        "npa.cli.workbench.workflow._preflight_submit_gang_capacity",
        side_effect=RuntimeError("completed generation no longer has free GPUs"),
    )
    mocker.patch("npa.cli.workbench.workflow._preflight_submit_images", return_value={})
    mocker.patch(
        "npa.cli.workbench.workflow._resolve_submit_accelerators", return_value={}
    )
    mocker.patch("npa.cli.workbench.workflow._adopt_npa_kubeconfig", return_value=True)
    mocker.patch("npa.cli.workbench.workflow._verify_submit_controller_owner")
    mocker.patch(
        "npa.orchestration.npa_workflow.model_cache_preflight.adopt_model_cache_claim",
        return_value="",
    )

    def target_preflight(_spec, **kwargs):
        if kwargs.get("gpu_check") is not None:
            kwargs["gpu_check"]()
        return None, {}

    target = mocker.patch(
        "npa.cli.workbench.workflow._execution_target_preflight",
        side_effect=target_preflight,
    )
    arguments = [
        "workbench",
        "workflow",
        "submit",
        str(gpu_then_cpu_spec),
        "--run-id",
        "wave-capacity",
        "--runtime",
        "--infra",
        "k8s/unit-context",
        "--no-deploy-if-absent",
    ]
    result = RUNNER.invoke(app, [*arguments, *(["--resume"] if resume else [])])
    assert result.exit_code == 0, result.output
    assert fake_runtime["options"].resume is resume
    target.assert_called_once()
    assert target.call_args.kwargs["verify_cluster"] is True
    wave = tmp_path / "publish.yaml"
    wave.write_text(
        "name: publish\nresources: {cloud: kubernetes, cpus: 4, memory: 16}\nrun: 'true'\n"
    )
    fake_runtime["options"].pre_submit_hook(wave)
    capacity.assert_not_called()


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


def test_runtime_automatically_uses_resolved_project_storage_credentials(
    mocker, monkeypatch, satisfied_preflight
) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        lambda **kwargs: type(
            "Context",
            (),
            {
                "endpoint_url": "https://storage.example.invalid",
                "access_key_id": "project-access",
                "secret_access_key": "project-secret",
                "secret_values": {},
                "missing": (),
            },
        )(),
    )
    observed: dict[str, object] = {}

    def fake_run(spec, **kwargs):  # noqa: ANN001
        observed["access"] = os.environ.get("AWS_ACCESS_KEY_ID")
        observed["secret"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        observed["secret_envs"] = kwargs["options"].secret_envs
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
            "rt-project-storage-creds",
            "--runtime",
            "--var",
            "bucket=rt-bucket",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "access": "project-access",
        "secret": "project-secret",
        "secret_envs": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    }
    assert "project-access" not in result.output
    assert "project-secret" not in result.output
    assert "AWS_ACCESS_KEY_ID" not in os.environ
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ


@pytest.fixture()
def configured_gpu_runtime(mocker, monkeypatch, satisfied_preflight):
    """Resolve private storage once and observe both actual discovery calls."""
    from npa.clients.config import StorageConfig
    from npa.orchestration.npa_workflow.submit_credentials import (
        STORAGE_ENDPOINT_ENV_NAMES,
    )

    expected = {
        "AWS_ACCESS_KEY_ID": "project-access",
        "AWS_SECRET_ACCESS_KEY": "project-secret",
        "NPA_S3_BUCKET": "rt-bucket",
        "NPA_S3_PREFIX": "runs/identity-test",
        **dict.fromkeys(STORAGE_ENDPOINT_ENV_NAMES, "https://storage.example.invalid"),
    }
    for key in expected:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://ambient.example.invalid")
    original_environment = {key: os.environ.get(key) for key in expected}
    mocker.patch(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage",
        return_value=StorageConfig(
            "rt-bucket",
            expected["AWS_ENDPOINT_URL"],
            expected["AWS_ACCESS_KEY_ID"],
            expected["AWS_SECRET_ACCESS_KEY"],
        ),
    )
    for name in ("_preflight_submit_images", "_verify_submit_controller_owner"):
        mocker.patch(f"npa.cli.workbench.workflow.{name}", return_value={})
    mocker.patch("npa.cli.workbench.workflow._adopt_npa_kubeconfig", return_value=True)
    mocker.patch(
        "npa.orchestration.npa_workflow.model_cache_preflight.adopt_model_cache_claim",
        return_value="",
    )
    mocker.patch("npa.orchestration.skypilot.workflow.ensure_local_api_daemon_health")
    observations = []

    def discovery(*args, **kwargs):
        observations.append({key: os.environ.get(key) for key in expected})
        return {}

    mocker.patch(
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators",
        side_effect=discovery,
    )
    return expected, observations, original_environment


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_runtime_discovery_preserves_resolved_identity_and_restores_environment(
    configured_gpu_runtime,
    gpu_then_cpu_spec: Path,
    mocker,
    runtime_failure: bool,
) -> None:
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError

    expected, observations, original_environment = configured_gpu_runtime
    wave = gpu_then_cpu_spec.with_name("wave.yaml")
    wave.write_text(
        "name: generate\nresources: {cloud: kubernetes, accelerators: 'B200:1'}\nrun: 'true'\n"
    )

    def run(spec, **kwargs):
        # Exercise the actual per-wave CLI hook under the runtime's environment.
        kwargs["options"].pre_submit_hook(wave)
        observations.append({key: os.environ.get(key) for key in expected})
        if runtime_failure:
            raise NpaWorkflowError("synthetic runtime failure")
        return RuntimeReport(
            workflow=spec.name, run_id=kwargs["run_id"], status="succeeded"
        )

    mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime", side_effect=run
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(gpu_then_cpu_spec),
            "--run-id",
            "identity-test",
            "--runtime",
            "--infra",
            "k8s/unit-context",
            "--var",
            "prefix=runs/{{run.id}}",
            "--no-deploy-if-absent",
            "--s3-endpoint",
            "https://storage.example.invalid",
        ],
    )

    assert result.exit_code == int(runtime_failure), result.output
    assert observations == [expected, expected, expected]
    assert {key: os.environ.get(key) for key in expected} == original_environment
    assert "project-access" not in result.output
    assert "project-secret" not in result.output


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


def test_runtime_required_workflow_selects_runtime_automatically(
    fake_runtime, tmp_path: Path
) -> None:
    runtime_spec = tmp_path / "runtime-required.yaml"
    runtime_spec.write_text(
        FANOUT.read_text(encoding="utf-8").replace(
            "metadata:\n", "metadata:\n  executionMode: runtime\n", 1
        ),
        encoding="utf-8",
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(runtime_spec),
            "--run-id",
            "paidf-runtime-required",
            "--var",
            "bucket=rt-bucket",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_runtime["spec"].metadata["executionMode"] == "runtime"


def test_runtime_required_workflow_rejects_explicit_no_runtime(mocker) -> None:
    runtime_driver = mocker.patch(
        "npa.orchestration.npa_workflow.runtime.run_workflow_runtime"
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(PAIDF_COSMOS3),
            "--run-id",
            "paidf-no-runtime",
            "--no-runtime",
        ],
    )

    assert result.exit_code == 1
    assert "requires runtime execution" in result.output
    runtime_driver.assert_not_called()


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
    mocker, monkeypatch, satisfied_preflight, tmp_path
) -> None:
    """The one-shot path recovers from the original declarative source spec."""

    operation_root = tmp_path / "operations"
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(operation_root))
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
        if submit_calls == 1:
            raise SkyPilotSubmitError(
                "synthetic indeterminate launch", launch_attempted=True
            )
        return WorkflowResult(
            status="SUBMITTED",
            job_id="9",
            returncode=0,
            launch_transaction={"state": "adopted"},
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

    assert result.exit_code == 1, result.output
    runtime_driver.assert_not_called()
    # The parallel group is flattened into today's serial pipeline.
    assert "execution: serial" in str(submitted["content"])
    assert "caption-shard-c" in str(submitted["content"])

    [journal_path] = operation_root.glob("*/journal.json")
    initial = json.loads(journal_path.read_text(encoding="utf-8"))
    recovery_argv = initial["recovery_commands"]["resume_argv"]
    assert Path(recovery_argv[4]) == FANOUT.resolve()
    assert Path(recovery_argv[4]).is_file()
    assert recovery_argv[recovery_argv.index("--resume-run") + 1] == "one-shot-1"
    assert "--no-runtime" in recovery_argv
    assert recovery_argv[recovery_argv.index("--var") + 1] == "bucket=rt-bucket"

    resumed = RUNNER.invoke(app, recovery_argv[1:])
    assert resumed.exit_code == 0, resumed.output
    assert "status: SUBMITTED" in resumed.output
    assert submit_mock.call_count == 2
    final = json.loads(journal_path.read_text(encoding="utf-8"))
    assert final["phase"] == "committed"
    assert final["resume_count"] == 1
    assert len(list(operation_root.glob("*/journal.json"))) == 1


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


@pytest.fixture()
def runtime_capacity_preflight(fake_runtime, mocker):
    """Reach actual CLI capacity ordering without contacting infrastructure."""
    from npa.cli.workbench import workflow as workflow_cli

    capacity = mocker.patch.object(
        workflow_cli,
        "_preflight_submit_gang_capacity",
        side_effect=RuntimeError("existing GPU job occupies available capacity"),
    )
    mocker.patch.object(workflow_cli, "_submit_prerequisites", return_value=[])
    mocker.patch.object(workflow_cli, "_verify_submit_controller_owner")
    mocker.patch.object(workflow_cli, "_adopt_npa_kubeconfig")
    mocker.patch(
        "npa.orchestration.npa_workflow.model_cache_preflight.adopt_model_cache_claim",
        return_value=None,
    )
    mocker.patch.object(workflow_cli, "_resolve_submit_accelerators", return_value={})
    mocker.patch.object(workflow_cli, "_preflight_submit_images", return_value={})

    def verify_target(*args, **kwargs):
        if kwargs["gpu_check"] is not None:
            kwargs["gpu_check"]()
        return None, {"scope": "pass"}

    target = mocker.patch.object(
        workflow_cli,
        "_execution_target_preflight",
        side_effect=verify_target,
    )
    return capacity, target


def _capacity_submit_args(*extra):
    return [
        "workbench",
        "workflow",
        "submit",
        str(FANOUT),
        "--infra",
        "k8s/unit-context",
        "--no-deploy-if-absent",
        "--var",
        "bucket=rt-bucket",
        *extra,
    ]


@pytest.mark.parametrize(
    "runtime_args,resume",
    [
        (["--run-id", "rt-capacity-fresh"], False),
        (["--resume-run", "rt-capacity-resume"], True),
        (["--run-id", "rt-capacity-resume", "--resume"], True),
    ],
)
def test_runtime_capacity_evidence_waits_for_the_sdk_wave_gate(
    fake_runtime,
    runtime_capacity_preflight,
    tmp_path,
    runtime_args,
    resume,
):
    capacity, target = runtime_capacity_preflight
    result = RUNNER.invoke(app, _capacity_submit_args("--runtime", *runtime_args))
    assert result.exit_code == 0, result.output
    capacity.assert_not_called()
    target.assert_called_once()
    assert target.call_args.kwargs["gpu_check"] is None
    options = fake_runtime["options"]
    assert options.resume is resume
    assert options.preflight_evidence["gang_capacity"] == "unknown"

    wave = tmp_path / "next-wave.yaml"
    wave.write_text("name: next-wave\nrun: echo next\n")
    options.pre_submit_hook(wave)
    options.pre_submit_hook(wave)
    assert options.preflight_evidence["gang_capacity"] == "unknown"
    capacity.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        ["--run-id", "rt-fresh-capacity"],
        ["--resume-run", "rt-one-shot-capacity"],
    ],
)
def test_one_shot_submit_still_requires_free_capacity(
    fake_runtime,
    runtime_capacity_preflight,
    extra,
):
    capacity, target = runtime_capacity_preflight
    result = RUNNER.invoke(app, _capacity_submit_args(*extra))
    assert result.exit_code == 1, result.output
    assert "existing GPU job occupies available capacity" in result.output
    capacity.assert_called_once()
    target.assert_called_once()
    assert target.call_args.kwargs["gpu_check"] is not None
    assert fake_runtime == {}


def test_runtime_resume_keeps_execution_scope_gate(
    fake_runtime,
    runtime_capacity_preflight,
):
    capacity, target = runtime_capacity_preflight
    target.side_effect = RuntimeError("execution scope mismatch")
    result = RUNNER.invoke(
        app,
        _capacity_submit_args(
            "--runtime",
            "--resume-run",
            "rt-scope-mismatch",
        ),
    )
    assert result.exit_code == 1, result.output
    assert "execution scope mismatch" in result.output
    capacity.assert_not_called()
    assert fake_runtime == {}


@pytest.fixture()
def runtime_api_environment(fake_runtime, mocker, monkeypatch, tmp_path):
    """Capture the real readiness/runtime boundary with only network calls faked."""
    from npa.orchestration.npa_workflow.submit_credentials import (
        STORAGE_ENDPOINT_ENV_NAMES,
        SubmitCredentialContext,
    )

    monkeypatch.setenv("NPA_S3_PREFIX", "inherited/old-run")
    monkeypatch.setenv("NPA_S3_BUCKET", "inherited-bucket")
    for name in STORAGE_ENDPOINT_ENV_NAMES:
        monkeypatch.setenv(name, "https://inherited.invalid")
    selected = {
        "AWS_ACCESS_KEY_ID": "selected-access",
        "AWS_SECRET_ACCESS_KEY": "selected-secret",
    }
    credentials = SubmitCredentialContext(
        endpoint_url="https://selected.invalid",
        secret_values=selected,
        access_key_id=selected["AWS_ACCESS_KEY_ID"],
        secret_access_key=selected["AWS_SECRET_ACCESS_KEY"],
    )
    mocker.patch(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        return_value=credentials,
    )
    mocker.patch(
        "npa.orchestration.skypilot.k8s_gpu_catalog.spec_accelerators",
        return_value={"L4": 1},
    )
    snapshots = []
    for target in (
        "npa.orchestration.skypilot.workflow.ensure_local_api_daemon_health",
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators",
    ):
        mocker.patch(
            target,
            side_effect=lambda *args, **kwargs: (
                snapshots.append(dict(os.environ)) or {}
            ),
        )
    driver = mocker.patch("npa.orchestration.npa_workflow.runtime.run_workflow_runtime")
    wave = tmp_path / "wave.yaml"
    wave.write_text("name: next-wave\nrun: echo next\n")

    def run(spec, **kwargs):
        snapshots.append(dict(os.environ))
        kwargs["options"].pre_submit_hook(wave)
        return RuntimeReport(
            workflow=spec.name, run_id=kwargs["run_id"], status="succeeded"
        )

    driver.side_effect = run
    names = ("NPA_S3_PREFIX", "NPA_S3_BUCKET", *STORAGE_ENDPOINT_ENV_NAMES, *selected)
    return snapshots, driver, names


@pytest.mark.parametrize(
    "prefix_args,ambient,expected",
    [
        (["--var", "prefix=selected/{{run.id}}"], True, "selected/environment-test"),
        (
            ["--var", "prefix=ignored", "--s3-prefix", "explicit/{{run.id}}"],
            True,
            "explicit/environment-test",
        ),
        ([], False, "token-factory-fanout/environment-test"),
        ([], True, "inherited/old-run"),
    ],
)
def test_runtime_readiness_uses_resolved_environment(
    runtime_api_environment,
    monkeypatch,
    prefix_args,
    ambient,
    expected,
):
    snapshots, driver, names = runtime_api_environment
    if not ambient:
        monkeypatch.delenv("NPA_S3_PREFIX")
    before = {name: os.environ.get(name) for name in names}
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--run-id",
            "environment-test",
            "--var",
            "bucket=selected-bucket",
            *prefix_args,
        ],
    )
    assert result.exit_code == 0, result.output
    assert driver.call_count == 1
    assert len(snapshots) == 5
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert snapshots[0]["NPA_S3_PREFIX"] == expected
    assert snapshots[0]["NPA_S3_BUCKET"] == "selected-bucket"
    assert snapshots[0]["AWS_ACCESS_KEY_ID"] == "selected-access"
    assert snapshots[0]["AWS_SECRET_ACCESS_KEY"] == "selected-secret"
    assert all(snapshots[0][name] == "https://selected.invalid" for name in names[2:-2])
    assert {name: os.environ.get(name) for name in names} == before


def test_runtime_readiness_uses_explicit_isolated_state_and_config(
    runtime_api_environment,
    tmp_path,
):
    snapshots, driver, _ = runtime_api_environment
    isolated = tmp_path / "isolated"
    config = tmp_path / "sky.yaml"
    config.write_text("kubernetes: {}\n", encoding="utf-8")

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--run-id",
            "isolated-environment-test",
            "--var",
            "bucket=selected-bucket",
            "--isolated-config-dir",
            str(isolated),
            "--config-path",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    assert driver.call_count == 1
    assert len(snapshots) == 5
    assert snapshots[0]["NPA_SKYPILOT_ISOLATED_CONFIG_DIR"] == str(isolated.resolve())
    assert snapshots[0]["SKYPILOT_GLOBAL_CONFIG"] == str(config.resolve())
    assert all(snapshot == snapshots[0] for snapshot in snapshots)


@pytest.mark.parametrize("boundary", ["readiness", "runtime"])
def test_runtime_environment_is_restored_after_failure(
    runtime_api_environment, mocker, boundary
):
    snapshots, driver, names = runtime_api_environment
    before = {name: os.environ.get(name) for name in names}
    target = (
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators"
        if boundary == "readiness"
        else "npa.orchestration.npa_workflow.runtime.run_workflow_runtime"
    )
    failure = mocker.patch(target, side_effect=ValueError("executing identity changed"))
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--run-id",
            "environment-failure",
            "--var",
            "bucket=selected-bucket",
            "--var",
            "prefix=selected/{{run.id}}",
        ],
    )
    assert result.exit_code != 0
    failure.assert_called_once()
    assert snapshots[0]["NPA_S3_PREFIX"] == "selected/environment-failure"
    assert {name: os.environ.get(name) for name in names} == before
    if boundary == "readiness":
        driver.assert_not_called()


def test_plan_only_does_not_bind_runtime_api_environment(runtime_api_environment):
    snapshots, driver, names = runtime_api_environment
    before = {name: os.environ.get(name) for name in names}
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--plan-only",
            "--run-id",
            "environment-plan",
            "--var",
            "bucket=selected-bucket",
            "--var",
            "prefix=selected/{{run.id}}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert snapshots == []
    driver.assert_not_called()
    assert {name: os.environ.get(name) for name in names} == before


def test_absolute_prefix_submission_receipt_matches_canonical_store(
    tmp_path: Path,
) -> None:
    from npa.cli.workbench.workflow import _workflow_submission_receipt
    from npa.orchestration.npa_workflow.run_state import store_for_config
    from npa.orchestration.npa_workflow.spec import load_spec

    source = tmp_path / "absolute-prefix.yaml"
    source.write_text(
        """apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: absolute-prefix}
config:
  bucket: unit-output
  prefix: s3://unit-output/task/unit-run
initial: execute
states:
  execute:
    run: {shell: 'true'}
    terminal: true
"""
    )
    spec = load_spec(source)
    receipt = _workflow_submission_receipt(spec, [], "unit-run")
    store = store_for_config(spec.config, run_id="unit-run")

    assert store is not None
    assert receipt["run_prefix_uri"] == store.run_prefix_uri
    assert receipt["manifest_uri"] == (
        "s3://unit-output/task/unit-run/npa-workflow/manifest.json"
    )


def test_resume_store_uses_exact_recorded_legacy_location(monkeypatch) -> None:
    from types import SimpleNamespace

    from npa.cli.workbench.workflow import _recorded_runtime_store
    from npa.orchestration.npa_workflow import submission_state

    legacy = "s3://unit-output/s3://unit-output/task/legacy-run"
    monkeypatch.setattr(
        submission_state,
        "inspect_submission_state",
        lambda project, run_id: SimpleNamespace(
            outcome="found",
            payload={"workflow": {"run_prefix_uri": legacy}},
        ),
    )

    store = _recorded_runtime_store(
        "unit", "legacy-run", resume=True, credentials=_selected_storage_credentials()
    )

    assert store is not None
    assert store.run_prefix_uri == legacy
    assert store.prefix == "s3://unit-output/task/legacy-run"
    assert store._endpoint_url == "https://storage.us-central1.nebius.cloud"
    assert store._aws_access_key_id == "selected-access"
    assert store._aws_secret_access_key == "selected-secret"


def test_fresh_runtime_does_not_reuse_recorded_location(monkeypatch) -> None:
    from npa.cli.workbench.workflow import _recorded_runtime_store
    from npa.orchestration.npa_workflow import submission_state

    called = False

    def inspect(project: str, run_id: str):
        nonlocal called
        called = True
        raise AssertionError("fresh runs must not inspect an old receipt")

    monkeypatch.setattr(submission_state, "inspect_submission_state", inspect)

    assert (
        _recorded_runtime_store(
            "unit",
            "fresh-run",
            resume=False,
            credentials=_selected_storage_credentials(),
        )
        is None
    )
    assert not called


def test_resume_store_allows_an_absent_local_receipt(monkeypatch) -> None:
    from types import SimpleNamespace

    from npa.cli.workbench.workflow import _recorded_runtime_store
    from npa.orchestration.npa_workflow import submission_state

    monkeypatch.setattr(
        submission_state,
        "inspect_submission_state",
        lambda project, run_id: SimpleNamespace(outcome="absent", payload={}),
    )

    assert (
        _recorded_runtime_store(
            "unit", "new", resume=True, credentials=_selected_storage_credentials()
        )
        is None
    )


def _write_local_receipt(monkeypatch, tmp_path: Path, run_id: str, raw: str) -> None:
    from npa.clients import config
    from npa.orchestration.npa_workflow.submission_state import submission_state_path

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.yaml")
    path = submission_state_path("unit", run_id)
    path.parent.mkdir(parents=True)
    path.write_text(raw, encoding="utf-8")


def test_cli_resume_preserves_recorded_legacy_store(
    fake_runtime, monkeypatch, tmp_path: Path
) -> None:
    from npa.orchestration.npa_workflow.submission_state import load_submission_state

    run_id = "legacy-run"
    legacy = "s3://unit-output/s3://unit-output/task/legacy-run"
    payload = {
        "schema_version": "npa.workflow.submission.v1",
        "project": "unit",
        "run_id": run_id,
        "workflow": {"run_prefix_uri": legacy},
    }
    _write_local_receipt(monkeypatch, tmp_path, run_id, json.dumps(payload))
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--resume",
            "--run-id",
            run_id,
            "--project",
            "unit",
            "--var",
            "bucket=rt-bucket",
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_runtime["state_store"].run_prefix_uri == legacy
    receipt = load_submission_state("unit", run_id)
    assert receipt["workflow"]["run_prefix_uri"] == legacy
    assert receipt["workflow"]["manifest_uri"] == f"{legacy}/npa-workflow/manifest.json"


def test_cli_denied_recorded_prefix_stops_before_update_or_launch(
    monkeypatch, mocker, satisfied_preflight, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from npa import execution_preflight
    from npa.cli.workbench import workflow as workflow_cli

    legacy = "s3://unit-output/s3://unit-output/task/denied-run"
    _write_local_receipt(
        monkeypatch, tmp_path, "denied-run", _resume_receipt("denied-run", legacy)
    )
    captured = {}

    def resolve(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        workflow_cli, "_execution_target_preflight", REAL_EXECUTION_TARGET_PREFLIGHT
    )
    monkeypatch.setattr(execution_preflight, "resolve_execution_target", resolve)
    monkeypatch.setattr(
        execution_preflight,
        "verify_execution_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("legacy denied")),
    )
    update = mocker.patch(
        "npa.orchestration.npa_workflow.submission_state.update_submission_state"
    )
    driver = mocker.patch("npa.orchestration.npa_workflow.runtime.run_workflow_runtime")
    result = _invoke_runtime_resume("denied-run")

    assert result.exit_code == 1 and "legacy denied" in result.output
    assert legacy + "/" in captured["output_uris"]
    assert any(uri.startswith("s3://rt-bucket/") for uri in captured["output_uris"])
    assert captured["output_kinds"][legacy + "/"] == "directory"
    update.assert_not_called()
    driver.assert_not_called()


def _resume_receipt(run_id: str, legacy: str) -> str:
    return json.dumps(
        {
            "schema_version": "npa.workflow.submission.v1",
            "project": "unit",
            "run_id": run_id,
            "workflow": {"run_prefix_uri": legacy},
        }
    )


def _invoke_runtime_resume(run_id: str):
    return RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--resume",
            "--run-id",
            run_id,
            "--project",
            "unit",
            "--var",
            "bucket=rt-bucket",
        ],
    )


@pytest.mark.parametrize("receipt_kind", ["unavailable", "missing-workflow"])
def test_cli_resume_rejects_unusable_receipt_before_update_or_launch(
    receipt_kind, monkeypatch, mocker, satisfied_preflight, tmp_path: Path
) -> None:
    run_id = f"resume-{receipt_kind}"
    payload = {
        "schema_version": "npa.workflow.submission.v1",
        "project": "unit",
        "run_id": run_id,
    }
    raw = "{" if receipt_kind == "unavailable" else json.dumps(payload)
    _write_local_receipt(monkeypatch, tmp_path, run_id, raw)
    update = mocker.patch(
        "npa.orchestration.npa_workflow.submission_state.update_submission_state"
    )
    driver = mocker.patch("npa.orchestration.npa_workflow.runtime.run_workflow_runtime")
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(FANOUT),
            "--runtime",
            "--resume",
            "--run-id",
            run_id,
            "--project",
            "unit",
            "--var",
            "bucket=rt-bucket",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "receipt" in result.output
    update.assert_not_called()
    driver.assert_not_called()
