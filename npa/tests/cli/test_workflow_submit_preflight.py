"""Pre-submit prerequisite checks, `--var` on plan/run-spec, and `stage-src`.

A first `npa workbench workflow submit` used to fail one prerequisite at a time
(no npa source, then no SkyPilot CLI, then a placeholder bucket), each as a
separate run, and there was no command to produce the npa source copy at all.
"""

from __future__ import annotations

from io import StringIO
import json
import os
from pathlib import Path

import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import workflow as workflow_cli

runner = CliRunner()

SPEC = (
    Path(__file__).resolve().parents[3]
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "physical-ai-data-factory.yaml"
)
COSMOS3_SPEC = (
    Path(__file__).resolve().parents[3]
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "paidf-cosmos3.yaml"
)
SIM2REAL_SPEC = (
    Path(__file__).resolve().parents[3]
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "sim2real.yaml"
)


@pytest.fixture(autouse=True)
def _no_ambient_src(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    from npa.clients import storage_validation
    from npa.clients.storage_validation import StorageProbeResult

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


def _submit(*args: str):
    return runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--run-id",
            "preflight-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--no-deploy-if-absent",
            *args,
        ],
    )


def _submit_cosmos3(*args: str):
    return runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(COSMOS3_SPEC),
            "--run-id",
            "paidf-cosmos3-preflight-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--no-deploy-if-absent",
            *args,
        ],
    )


def test_fail_reports_bracketed_exception_messages_literally(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        workflow_cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )

    with pytest.raises(typer.Exit) as exc_info:
        workflow_cli._fail("invalid target [H100:1] after closing tag [/:]")

    assert exc_info.value.exit_code == 1
    assert output.getvalue() == (
        "Error: invalid target [H100:1] after closing tag [/:]\n"
    )


def test_submit_lists_every_missing_prerequisite_at_once() -> None:
    result = _submit()

    assert result.exit_code == 1, result.output
    assert "missing prerequisites" in result.output
    # Source is staged automatically; runtime and bucket blockers are still
    # reported together.
    assert "SkyPilot CLI is not usable" in result.output
    assert "npa skypilot bootstrap" in result.output
    assert "NPA_SRC_S3_URI is unset" not in result.output
    assert "example-bucket" in result.output
    assert "--var bucket=<your-bucket>" in result.output
    assert "--skip-preflight" in result.output


def test_submit_preflight_does_not_reach_skypilot(mocker) -> None:
    submit_workflow = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow"
    )

    result = _submit()

    assert result.exit_code == 1
    submit_workflow.assert_not_called()


def test_sim2real_submit_collects_pipeline_prerequisites_before_image_or_launch(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    from subprocess import CompletedProcess

    image_preflight = mocker.patch(
        "npa.cli.workbench.workflow._preflight_submit_images"
    )
    launch = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")
    monkeypatch.setattr(
        "npa.clients.kube.run_kubectl",
        lambda *args, **kwargs: CompletedProcess(args, 1, stdout="", stderr="NotFound"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SIM2REAL_SPEC),
            "--run-id",
            "sim2real-cold-start",
            "--no-deploy-if-absent",
            "--var",
            "bucket=real-bucket",
        ],
    )

    assert result.exit_code == 1
    assert "missing prerequisites" in result.output
    assert "controller_image" in result.output
    assert "AWS_ACCESS_KEY_ID" in result.output
    assert "no Ready" not in result.output  # node listing itself failed
    assert "Kubernetes nodes cannot be listed" in result.output
    assert "config.isaac_cache_pvc is empty" in result.output
    image_preflight.assert_not_called()
    launch.assert_not_called()


def test_paidf_submit_collects_runtime_prerequisites_before_image_or_launch(
    mocker,
) -> None:
    image_preflight = mocker.patch(
        "npa.cli.workbench.workflow._preflight_submit_images"
    )
    launch = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--run-id",
            "paidf-cold-start",
            "--no-deploy-if-absent",
            "--var",
            "bucket=real-bucket",
        ],
    )

    assert result.exit_code == 1
    assert "missing prerequisites" in result.output
    assert "PAIDF runtime credentials" in result.output
    assert "NEBIUS_TOKEN_FACTORY_KEY" in result.output
    assert "HF_TOKEN" in result.output
    image_preflight.assert_not_called()
    launch.assert_not_called()


def test_paidf_kubernetes_helper_propagates_context_and_kubeconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from subprocess import CompletedProcess

    from npa.cli.workbench.workflow import (
        _paidf_kubernetes_prerequisites_for_submit,
    )

    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return CompletedProcess(args, 1, stdout="", stderr="Forbidden")

    monkeypatch.setenv("KUBECONFIG", "/tmp/review-kubeconfig")
    monkeypatch.setattr("npa.clients.kube.run_kubectl", run)

    issues = _paidf_kubernetes_prerequisites_for_submit("paidf-review")

    assert issues
    assert calls == [
        (
            ["get", "nodes", "-o", "json"],
            {
                "context": "paidf-review",
                "kubeconfig": "/tmp/review-kubeconfig",
                "timeout": 30,
            },
        )
    ]


def test_paidf_placement_fails_before_storage_or_staging_without_explicit_infra(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    for name in (
        "NEBIUS_TOKEN_FACTORY_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.setenv(name, "redacted")
    monkeypatch.setenv("NPA_SKYPILOT_BIN", "/bin/true")
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._available_kube_contexts",
        lambda: ["npa-cluster"],
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._adopt_npa_kubeconfig", lambda _context: True
    )
    monkeypatch.setattr(
        "npa.controller_ownership.verify_controller_owner", lambda *_args: None
    )
    placement = mocker.patch(
        "npa.cli.workbench.workflow._paidf_kubernetes_prerequisites_for_submit",
        return_value=[("placement blocked", "resize the selected node")],
    )
    exact_access = mocker.patch(
        "npa.workbench.cosmos.checkpoint_access.preflight_control_checkpoint_access"
    )
    mocker.patch(
        "npa.cli.workbench.workflow._preflight_submit_images", return_value={}
    )
    storage = mocker.patch("npa.clients.storage_validation.probe_storage_write")
    prepare_input = mocker.patch(
        "npa.workflows.data_factory_input.prepare_paidf_input"
    )
    stage_source = mocker.patch(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source"
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--run-id",
            "paidf-placement-order",
            "--no-deploy-if-absent",
            "--var",
            "bucket=real-bucket",
            "--assume-decision",
            "promote_checkpoint",
            "--secret-env",
            "NEBIUS_TOKEN_FACTORY_KEY",
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--secret-env",
            "HF_TOKEN",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "placement blocked" in result.output
    placement.assert_called_once_with("npa-cluster")
    exact_access.assert_not_called()
    storage.assert_not_called()
    prepare_input.assert_not_called()
    stage_source.assert_not_called()


def test_paidf_existing_target_orders_placement_exact_access_then_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    for name in (
        "NEBIUS_TOKEN_FACTORY_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.setenv(name, "redacted")
    monkeypatch.setenv("NPA_SKYPILOT_BIN", "/bin/true")
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._available_kube_contexts",
        lambda: ["npa-cluster"],
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._adopt_npa_kubeconfig", lambda _context: True
    )
    monkeypatch.setattr(
        "npa.clients.huggingface.validate_hf_access",
        lambda *_args, **_kwargs: pytest.fail(
            "broad repository-level Hugging Face probe must not run"
        ),
    )

    def placement(_context: str):
        events.append("placement")
        return []

    def exact_access(*, modality: str, token: str):
        assert token == "redacted"
        events.append(f"exact:{modality}")
        return {"status_code": 302}

    def image_preflight(*_args, **_kwargs):
        events.append("image")
        raise RuntimeError("stop after ordered image boundary")

    monkeypatch.setattr(
        "npa.cli.workbench.workflow._paidf_kubernetes_prerequisites_for_submit",
        placement,
    )
    monkeypatch.setattr(
        "npa.workbench.cosmos.checkpoint_access.preflight_control_checkpoint_access",
        exact_access,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._preflight_submit_images", image_preflight
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--run-id",
            "paidf-model-order",
            "--no-deploy-if-absent",
            "--var",
            "bucket=real-bucket",
            "--assume-decision",
            "promote_checkpoint",
            "--secret-env",
            "NEBIUS_TOKEN_FACTORY_KEY",
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--secret-env",
            "HF_TOKEN",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert events == ["placement", "exact:edge", "image"]


def test_sim2real_submit_propagates_explicit_kubernetes_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from subprocess import CompletedProcess

    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return CompletedProcess(args, 1, stdout="", stderr="NotFound")

    monkeypatch.setenv("KUBECONFIG", "/tmp/sim2real-kubeconfig")
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._adopt_npa_kubeconfig", lambda _context: True
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._available_kube_contexts",
        lambda: ["sim2real-review"],
    )
    monkeypatch.setattr("npa.clients.kube.run_kubectl", run)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SIM2REAL_SPEC),
            "--run-id",
            "sim2real-context",
            "--no-deploy-if-absent",
            "--infra",
            "k8s/sim2real-review",
            "--var",
            "bucket=real-bucket",
        ],
    )

    assert result.exit_code == 1
    assert calls
    assert all(call[1]["context"] == "sim2real-review" for call in calls)
    assert all(
        call[1]["kubeconfig"] == "/tmp/sim2real-kubeconfig" for call in calls
    )


def test_submit_preflight_clears_as_prerequisites_are_met(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each satisfied prerequisite drops out of the report."""
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")
    result = _submit("--var", "bucket=real-bucket")
    assert result.exit_code == 1
    assert "NPA_SRC_S3_URI is unset" not in result.output
    assert "example-bucket" not in result.output
    assert "SkyPilot CLI is not usable" in result.output

    # ... and with a resolvable sky binary the preflight passes entirely.
    sky = tmp_path / "sky"
    sky.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sky.chmod(0o755)
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(sky))
    result = _submit("--var", "bucket=real-bucket", "--plan-only")
    assert result.exit_code == 0, result.output
    assert "missing prerequisites" not in result.output


def test_plan_only_skips_runtime_only_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--plan-only` renders locally, so it must not demand a SkyPilot CLI."""
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")

    result = _submit("--plan-only")

    assert result.exit_code == 0, result.output
    assert "status: PLANNED" in result.output
    # The placeholder bucket is still surfaced, as a warning not a blocker.
    assert "example-bucket" in result.output


def test_plan_only_without_source_uri_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    from npa.orchestration.npa_workflow import first_run_state

    state_root = tmp_path / "workflow-runs"
    monkeypatch.setattr(first_run_state, "DEFAULT_ROOT", state_root)
    stage = mocker.patch("npa.orchestration.npa_workflow.src_staging.stage_npa_source")
    upload_input = mocker.patch("npa.workflows.data_factory_input.prepare_paidf_input")

    result = _submit("--plan-only", "--var", "bucket=real-bucket")

    assert result.exit_code == 0, result.output
    assert "source: planned (s3://real-bucket/npa-src/npa/" in result.output
    assert "submission_state: NOT_SUBMITTED" in result.output
    assert not state_root.exists()
    stage.assert_not_called()
    upload_input.assert_not_called()


def test_plan_only_human_output_is_compact_and_details_are_explicit() -> None:
    compact = _submit("--plan-only", "--var", "bucket=real-bucket")
    verbose = _submit("--plan-only", "--details", "--var", "bucket=real-bucket")

    assert compact.exit_code == 0, compact.output
    assert compact.output.count("setup:\n") == 1
    assert "stages:\n  1. generate-configs:" in compact.output
    assert "--- full rendered SkyPilot YAML ---" not in compact.output
    assert "details: pass --details" in compact.output
    assert verbose.exit_code == 0, verbose.output
    assert verbose.output.count("setup:\n") == 1
    assert "--- full rendered SkyPilot YAML ---" in verbose.output
    assert "name: physical-ai-data-factory" in verbose.output


def test_plan_only_json_retains_stable_full_details() -> None:
    result = _submit(
        "--plan-only",
        "--var",
        "bucket=real-bucket",
        "--output-format",
        "json",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert list(payload) == sorted(payload)
    assert payload["lifecycle_state"] == "PLAN_ONLY"
    assert payload["submission_state"] == "NOT_SUBMITTED"
    assert payload["submission_receipt"] is None
    assert payload["preflight"]["decision"] == "unknown"
    assert {
        item["name"]: item["status"] for item in payload["preflight"]["checks"]
    } == {
        "credentials": "ready",
        "writable_storage": "unknown",
        "source_staging": "ready",
    }
    assert payload["source"]["status"] == "planned"
    assert len(payload["plan"]["steps"]) == payload["steps"]
    assert "setup:" in payload["skypilot_yaml"]


def test_plan_only_never_labels_a_known_credential_blocker_ready() -> None:
    result = _submit(
        "--plan-only",
        "--var",
        "bucket=real-bucket",
        "--secret-env",
        "KNOWN_MISSING_TEST_SECRET",
        "--output-format",
        "json",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["preflight"]["decision"] == "blocked"
    credential = next(
        item for item in payload["preflight"]["checks"] if item["name"] == "credentials"
    )
    assert credential["status"] == "blocked"
    assert "KNOWN_MISSING_TEST_SECRET" in credential["reason"]


def test_paidf_input_selectors_conflict_before_preflight() -> None:
    result = _submit(
        "--plan-only",
        "--input-video",
        "local.mp4",
        "--input-uri",
        "s3://source-bucket/input.mp4",
    )

    assert result.exit_code == 1
    assert "options conflict" in result.output
    assert "missing prerequisites" not in result.output


def test_paidf_lerobot_selector_is_planned_without_object_store_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")

    result = _submit(
        "--plan-only",
        "--lerobot-uri",
        "s3://source-bucket/datasets/robot-run/",
        "--lerobot-camera",
        "observation.images.front",
        "--lerobot-episode",
        "3",
        "--require-explicit-lerobot-selection",
        "--var",
        "bucket=real-bucket",
        "--output-format",
        "json",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["lifecycle_state"] == "PLAN_ONLY"
    assert "Operator-supplied LeRobotDataset" not in result.output
    assert "input_source_format" not in result.output  # metadata, not an argv shim


def test_cosmos3_paidf_lerobot_selector_uses_the_real_input_preparer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")

    result = _submit_cosmos3(
        "--plan-only",
        "--infra",
        "k8s/test-context",
        "--lerobot-uri",
        "s3://source-bucket/datasets/robot-run/",
        "--lerobot-camera",
        "observation.images.cam_high",
        "--lerobot-episode",
        "0",
        "--require-explicit-lerobot-selection",
        "--var",
        "bucket=real-bucket",
        "--output-format",
        "json",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    prepare = next(
        step for step in payload["plan"]["steps"] if step["state"] == "prepare-input"
    )
    assert "s3://source-bucket/datasets/robot-run/" in prepare["argv"]
    assert "observation.images.cam_high" in prepare["argv"]


@pytest.mark.parametrize(
    ("args", "missing"),
    [
        (
            (
                "--lerobot-uri",
                "s3://source-bucket/datasets/robot-run/",
                "--lerobot-episode",
                "0",
            ),
            "--lerobot-camera",
        ),
        (
            (
                "--lerobot-uri",
                "s3://source-bucket/datasets/robot-run/",
                "--lerobot-camera",
                "observation.images.front",
            ),
            "--lerobot-episode",
        ),
    ],
)
def test_paidf_lerobot_strict_selector_fails_before_preflight(
    args: tuple[str, ...], missing: str
) -> None:
    result = _submit(
        "--plan-only",
        *args,
        "--require-explicit-lerobot-selection",
    )

    assert result.exit_code == 1
    assert missing in result.output
    assert "fails closed" in result.output
    assert "missing prerequisites" not in result.output


def test_paidf_lerobot_strict_selector_requires_dataset_uri() -> None:
    result = _submit(
        "--plan-only",
        "--require-explicit-lerobot-selection",
    )

    assert result.exit_code == 1
    assert "requires --lerobot-uri" in result.output
    assert "missing prerequisites" not in result.output


def test_paidf_lerobot_only_selectors_fail_without_dataset_uri() -> None:
    result = _submit("--plan-only", "--lerobot-camera", "front")

    assert result.exit_code == 1
    assert "require --lerobot-uri" in result.output
    assert "missing prerequisites" not in result.output


def test_paidf_fixture_is_explicit_in_rendered_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")

    result = _submit(
        "--plan-only",
        "--seed-fixture",
        "--var",
        "bucket=real-bucket",
        "--output-format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert "Synthetic seeded fixture" not in result.output  # metadata, not a fake stage
    assert "generate_configs" in result.output
    plan = json.loads(result.output)["plan"]
    generate = next(
        step for step in plan["steps"] if step["state"] == "generate-configs"
    )
    assert generate["argv"][-4] == "true"
    assert generate["argv"][-1] == ""
    assert "--condition-on-input" in result.output


def test_skip_preflight_bypasses_the_checks(mocker) -> None:
    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=AssertionError("submit reached"),
    )

    result = _submit("--skip-preflight")

    # Not the preflight error: the run got past it (and then failed later).
    assert "missing prerequisites" not in result.output


@pytest.mark.parametrize(
    ("code", "summary"),
    [
        (
            "missing_configuration",
            "Writable S3 is not configured; missing AWS credentials.",
        ),
        ("bucket_unreachable", "S3 write probe could not find the configured bucket."),
        ("forbidden", "S3 write probe was forbidden."),
        ("cleanup_failed", "S3 probe cleanup failed; a temporary object remains."),
    ],
)
def test_s3_workflow_requires_a_successful_cleaned_write_probe(
    monkeypatch, code, summary
) -> None:
    from npa.cli.workbench.workflow import _submit_prerequisites
    from npa.clients import storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(False, code, summary),
    )

    missing = _submit_prerequisites(
        {"bucket": "real-bucket"},
        sky_bin="/bin/true",
        image="registry.example/npa-tool:v1",
        plan_only=False,
        requires_s3=True,
        s3_endpoint="https://storage.example",
        s3_access_key_id="access",
        s3_secret_access_key="secret",
    )

    assert any(summary in item for item, _remedy in missing)
    assert any("provision-if-absent" in remedy for _item, remedy in missing)


def test_non_s3_workflow_does_not_probe_or_require_storage(monkeypatch) -> None:
    from npa.cli.workbench.workflow import _submit_prerequisites
    from npa.clients import storage_validation

    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a non-S3 workflow must not probe storage")
        ),
    )

    missing = _submit_prerequisites(
        {},
        sky_bin="/bin/true",
        image="registry.example/npa-tool:v1",
        plan_only=False,
        requires_s3=False,
    )

    assert missing == []


def test_storage_requirement_is_derived_from_the_workflow_contract(tmp_path) -> None:
    from npa.cli.workbench.workflow import _spec_requires_s3

    s3_spec = tmp_path / "s3.yaml"
    s3_spec.write_text("config:\n  output: s3://{{config.bucket}}/results/\n")
    local_spec = tmp_path / "local.yaml"
    local_spec.write_text(
        "config:\n  bucket: local-directory\n  output: /tmp/results\n"
    )

    assert _spec_requires_s3(s3_spec) is True
    assert _spec_requires_s3(local_spec) is False


def test_image_override_satisfies_the_npa_source_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _submit("--image", "cr.example.invalid/reg/npa-tool:v1", "--plan-only")

    assert "NPA_SRC_S3_URI is unset" not in result.output


def test_config_pinned_resource_images_satisfy_the_npa_source_requirement() -> None:
    """Submit preflight must inspect the same ``--var`` config as rendering."""
    from npa.cli.workbench.workflow import _plan_requires_npa_source
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

    digest_image = f"cr.example.invalid/npa@sha256:{'a' * 64}"
    image_vars = {
        name: digest_image
        for name in (
            "controller_image",
            "transfer_image",
            "envgen_image",
            "reason_image",
            "isaac_image",
            "viewer_image",
        )
    }

    assert (
        _plan_requires_npa_source(
            SIM2REAL_SPEC,
            run_id="config-pinned-images",
            assume_decision="promote_checkpoint",
            config_overrides=image_vars,
            options=SkypilotRenderOptions(materialize_registry_secrets=False),
        )
        is False
    )


def test_preflight_images_accepts_the_same_config_vars_as_submit(mocker) -> None:
    """An empty canonical image input must be overridable before pull probes."""
    digest_image = f"cr.example.invalid/npa@sha256:{'a' * 64}"
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )
    args = [
        "workbench",
        "workflow",
        "preflight-images",
        str(SIM2REAL_SPEC),
        "--assume-decision",
        "promote_checkpoint",
    ]
    for name in (
        "controller_image",
        "transfer_image",
        "envgen_image",
        "reason_image",
        "isaac_image",
        "viewer_image",
    ):
        args.extend(["--var", f"{name}={digest_image}"])

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    checked_images = checks.call_args.args[0]
    assert checked_images
    assert set(checked_images) == {digest_image}


def test_image_none_automatically_plans_npa_source_staging() -> None:
    """`--image none` uses the automatic documented source-staging path."""
    result = _submit("--image", "none")

    assert result.exit_code == 1
    assert "NPA_SRC_S3_URI is unset" not in result.output


def test_plan_spec_var_overrides_the_placeholder_bucket() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "plan-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--var",
            "bucket=my-real-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    rendered = json.dumps(payload)
    assert "my-real-bucket" in rendered
    assert "example-bucket" not in rendered


def test_plan_spec_without_var_warns_about_the_placeholder() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "plan-demo",
            "--assume-decision",
            "promote_checkpoint",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "config.bucket is 'example-bucket'" in result.output
    assert "--var bucket=<your-bucket>" in result.output


def test_plan_spec_json_output_stays_machine_readable() -> None:
    """`--json` must emit a clean document, not the placeholder warning."""
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "plan-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "config.bucket is" not in result.output
    payload = json.loads(result.output)  # parses even with stderr mixed in
    # The prose warning is suppressed here, so the document has to carry the fact:
    # a plan against `example-bucket` looks valid but points at nothing.
    assert payload["bucket_is_placeholder"] is True


def test_plan_spec_json_omits_the_placeholder_flag_with_a_real_bucket() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--run-id",
            "plan-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--var",
            "bucket=real-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "bucket_is_placeholder" not in json.loads(result.output)


def test_run_spec_accepts_var_overrides() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(SPEC),
            "--run-id",
            "run-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--var",
            "bucket=my-real-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "my-real-bucket" in result.stdout
    assert "example-bucket" not in result.stdout


# ── --infra kube-context preflight ────────────────────────────────────────

from npa.cli.workbench.workflow import (  # noqa: E402
    _available_kube_contexts,
    _infra_kube_context,
    _submit_prerequisites,
)


def _write_kubeconfig(tmp_path: Path, *contexts: str) -> Path:
    import yaml

    path = tmp_path / "kubeconfig"
    path.write_text(yaml.safe_dump({"contexts": [{"name": c} for c in contexts]}))
    return path


def _prereq_items(**kwargs) -> list[str]:
    return [item for item, _remedy in _submit_prerequisites(**kwargs)]


def _mock_sky_bin_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.orchestration.skypilot._bin as skybin

    monkeypatch.setattr(skybin, "resolve_sky_bin", lambda _b: "/usr/bin/sky")


# A registry-pinned image satisfies the npa-source requirement, isolating the
# kube-context check.
_PINNED_IMAGE = "cr.eu-north1.nebius.cloud/reg/npa-lerobot:tag"


def test_infra_kube_context_extracts_only_a_pinned_k8s_context() -> None:
    assert _infra_kube_context("k8s/prod") == "prod"
    assert _infra_kube_context("kubernetes/np-cluster") == "np-cluster"
    assert _infra_kube_context("k8s") == ""  # no pinned context
    assert _infra_kube_context("nebius") == ""  # non-k8s target
    assert _infra_kube_context("") == ""


def test_available_kube_contexts_none_when_unreadable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "does-not-exist"))
    assert _available_kube_contexts() is None


def test_available_kube_contexts_reads_names(monkeypatch, tmp_path) -> None:
    kubeconfig = _write_kubeconfig(tmp_path, "alpha", "beta")
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    assert _available_kube_contexts() == ["alpha", "beta"]


def test_submit_preflight_flags_a_missing_kube_context(monkeypatch, tmp_path) -> None:
    _mock_sky_bin_ok(monkeypatch)
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "other-ctx")))
    items = _prereq_items(
        spec_config={"bucket": "real-bucket"},
        sky_bin="",
        image=_PINNED_IMAGE,
        plan_only=False,
        infra="k8s/missing-ctx",
    )
    assert any("kube context 'missing-ctx'" in item for item in items)
    assert any("other-ctx" in item for item in items)  # lists what is available


def test_submit_preflight_accepts_a_present_kube_context(monkeypatch, tmp_path) -> None:
    _mock_sky_bin_ok(monkeypatch)
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "prod")))
    items = _prereq_items(
        spec_config={"bucket": "real-bucket"},
        sky_bin="",
        image=_PINNED_IMAGE,
        plan_only=False,
        infra="k8s/prod",
    )
    assert not any("kube context" in item for item in items)


def test_submit_preflight_skips_context_check_when_kubeconfig_unreadable(
    monkeypatch, tmp_path
) -> None:
    _mock_sky_bin_ok(monkeypatch)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "does-not-exist"))
    items = _prereq_items(
        spec_config={"bucket": "real-bucket"},
        sky_bin="",
        image=_PINNED_IMAGE,
        plan_only=False,
        infra="k8s/anything",
    )
    assert not any("kube context" in item for item in items)


def test_plan_only_skips_the_kube_context_check(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "other-ctx")))
    items = _prereq_items(
        spec_config={"bucket": "real-bucket"},
        sky_bin="",
        image=_PINNED_IMAGE,
        plan_only=True,
        infra="k8s/missing-ctx",
    )
    assert not any("kube context" in item for item in items)


# ── the context check must not block the flag that creates the context ────────

HARDENING_SPEC = (
    Path(__file__).resolve().parents[3]
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "adversarial-scenario-hardening.yaml"
)


def test_deploy_if_absent_quota_blocker_precedes_all_submit_mutation(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    from npa.provisioning_preflight import PreflightBlockedError

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")
    plan = mocker.patch(
        "npa.orchestration.npa_workflow.deploy.plan_infra_present",
        side_effect=PreflightBlockedError(
            "compute.instance.count required=2 available=0 shortfall=2"
        ),
    )
    storage = mocker.patch(
        "npa.cli.workbench.workflow._submit_prerequisites",
        side_effect=AssertionError("storage preflight reached after quota blocker"),
    )
    images = mocker.patch("npa.cli.workbench.workflow._preflight_submit_images")
    stage = mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit")
    ensure = mocker.patch("npa.orchestration.npa_workflow.deploy.ensure_infra_present")
    launch = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(HARDENING_SPEC),
            "--run-id",
            "blocked-before-mutation",
            "--infra",
            "k8s/npa-cluster",
            "--accept-eula",
            "--var",
            "bucket=real-bucket",
        ],
    )

    assert result.exit_code == 1
    assert "compute.instance.count" in result.output
    plan.assert_called_once()
    storage.assert_not_called()
    images.assert_not_called()
    stage.assert_not_called()
    ensure.assert_not_called()
    launch.assert_not_called()


def test_submit_preflight_skips_context_check_for_self_provisioning_specs(
    monkeypatch, tmp_path
) -> None:
    """--deploy-if-absent creates the context, so its absence is the normal start."""
    _mock_sky_bin_ok(monkeypatch)
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "other-ctx")))
    items = _prereq_items(
        spec_config={"bucket": "real-bucket"},
        sky_bin="",
        image=_PINNED_IMAGE,
        plan_only=False,
        infra="k8s/npa-cluster",
        self_provisions=True,
    )
    assert not any("kube context" in item for item in items)


def test_spec_self_provisions_detects_deploy_if_absent_targets(tmp_path) -> None:
    from npa.cli.workbench.workflow import _spec_self_provisions

    assert _spec_self_provisions(HARDENING_SPEC) is True

    plain = tmp_path / "plain.yaml"
    plain.write_text(
        "apiVersion: npa.workflow/v0.0.1\n"
        "kind: Workflow\n"
        "resources:\n"
        "  gpu:\n"
        "    cloud: kubernetes\n"
        "    accelerators: RTXPRO6000:1\n",
        encoding="utf-8",
    )
    assert _spec_self_provisions(plain) is False
    # An unreadable spec never turns the preflight itself into the failure.
    assert _spec_self_provisions(Path("/nonexistent/spec.yaml")) is False


def test_data_factory_blueprint_provisions_from_the_spec_not_a_command() -> None:
    """The blueprint chains provisioning through YAML, so no `paidf up` verb is needed."""
    from npa.cli.workbench.workflow import _spec_self_provisions

    assert _spec_self_provisions(SPEC) is True


def test_adopt_npa_kubeconfig_points_kubeconfig_at_the_provisioned_cluster(
    monkeypatch, tmp_path
) -> None:
    """npa keeps cluster kubeconfigs out of ~/.kube/config, so submit must find them.

    Regression: after `npa provision-if-absent` created `npa-cluster`, a submit
    with `--infra k8s/npa-cluster` still failed with `Context npa-cluster not
    found ... Available contexts: []` because the kubeconfig npa wrote was never
    put on KUBECONFIG.
    """
    import yaml

    from npa.cli.workbench.workflow import _adopt_npa_kubeconfig
    from npa.cluster import state as state_module

    clusters = tmp_path / "clusters"
    npa_kubeconfig = clusters / "npa-cluster" / "kubeconfig"
    npa_kubeconfig.parent.mkdir(parents=True)
    npa_kubeconfig.write_text(yaml.safe_dump({"contexts": [{"name": "npa-cluster"}]}))
    monkeypatch.setattr(state_module, "CLUSTERS_DIR", clusters)
    other = _write_kubeconfig(tmp_path, "other-ctx")
    monkeypatch.setenv("KUBECONFIG", str(other))

    assert _adopt_npa_kubeconfig("npa-cluster") is True
    entries = os.environ["KUBECONFIG"].split(os.pathsep)
    assert entries[0] == str(npa_kubeconfig)
    assert str(other) in entries  # the operator's own contexts stay resolvable
    assert _available_kube_contexts() == ["npa-cluster", "other-ctx"]


def test_adopt_npa_kubeconfig_reports_a_context_npa_cannot_resolve(
    monkeypatch, tmp_path
) -> None:
    from npa.cli.workbench.workflow import _adopt_npa_kubeconfig
    from npa.cluster import state as state_module

    monkeypatch.setattr(state_module, "CLUSTERS_DIR", tmp_path / "clusters")
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "other-ctx")))

    assert _adopt_npa_kubeconfig("npa-cluster") is False
    # An already-visible context needs no adoption.
    assert _adopt_npa_kubeconfig("other-ctx") is True


def test_submit_fails_clearly_when_provisioning_left_no_context(
    monkeypatch, tmp_path, mocker
) -> None:
    """A `partial` provision used to hand the failure to `sky jobs launch`."""
    from npa.cluster import state as state_module

    monkeypatch.setattr(state_module, "CLUSTERS_DIR", tmp_path / "clusters")
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "other-ctx")))
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")
    _mock_sky_bin_ok(monkeypatch)
    mocker.patch(
        "npa.orchestration.npa_workflow.deploy.ensure_infra_present",
        return_value=[
            {
                "profile": "adversary-gpu",
                "cluster_name": "npa-cluster",
                "context": "npa-cluster",
                "accelerators": "RTXPRO6000:1",
                "status": "partial",
                "actions": ["s3:skipped"],
                "warnings": [
                    "project_id and tenant_id are required to ensure Kubernetes"
                ],
                "dry_run": False,
            }
        ],
    )
    mocker.patch(
        "npa.orchestration.npa_workflow.deploy.plan_infra_present",
        return_value={"npa-cluster": mocker.Mock()},
    )
    # Registry pull semantics are covered independently; this test reaches the
    # post-provision context diagnostic.
    mocker.patch("npa.cli.workbench.workflow._preflight_submit_images")
    launched = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(HARDENING_SPEC),
            "--run-id",
            "no-context-demo",
            "--infra",
            "k8s/npa-cluster",
            "--accept-eula",
            "--var",
            "bucket=real-bucket",
        ],
    )

    assert result.exit_code != 0
    # The provisioning warning is no longer swallowed ...
    assert "project_id and tenant_id are required" in result.output
    # ... and the submit stops with the remedy instead of launching. (Rich wraps
    # the message, so match fragments that survive a line break.)
    assert "Kube context" in result.output
    assert "provision-if-absent" in result.output
    assert not launched.called


def test_submit_lets_a_deploy_if_absent_spec_provision_its_own_context(
    monkeypatch, tmp_path, mocker
) -> None:
    """The preflight used to reject the context that --deploy-if-absent creates."""
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "other-ctx")))
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")
    _mock_sky_bin_ok(monkeypatch)
    planned_targets = []
    provisioned_targets = []

    def plan(targets, *, mutation):
        assert mutation is True
        planned_targets.extend(targets)
        return {"submit-context": mocker.Mock()}

    def ensure(targets, **_kwargs):
        provisioned_targets.extend(targets)
        return []

    ensure_infra_present = mocker.patch(
        "npa.orchestration.npa_workflow.deploy.ensure_infra_present",
        side_effect=ensure,
    )
    mocker.patch(
        "npa.orchestration.npa_workflow.deploy.plan_infra_present",
        side_effect=plan,
    )
    # Registry pull semantics are covered independently; this test reaches the
    # deploy-if-absent delegation path.
    mocker.patch("npa.cli.workbench.workflow._preflight_submit_images")
    mocker.patch(
        "npa.orchestration.npa_workflow.submit.prepare_npa_workflow_for_submit",
        side_effect=RuntimeError("stop after deployIfAbsent"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(HARDENING_SPEC),
            "--run-id",
            "self-provision-demo",
            "--project",
            "submit-project",
            "--infra",
            "k8s/submit-context",
            "--accept-eula",
            "--var",
            "bucket=real-bucket",
        ],
    )

    assert "kube context" not in result.output
    assert ensure_infra_present.called
    assert planned_targets == provisioned_targets
    assert planned_targets
    assert all(target.project == "submit-project" for target in planned_targets)
    assert all(target.cluster_name == "submit-context" for target in planned_targets)
    assert all(target.context == "submit-context" for target in planned_targets)
