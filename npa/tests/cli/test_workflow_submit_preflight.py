"""Pre-submit prerequisite checks, `--var` on plan/run-spec, and `stage-src`.

A first `npa workbench workflow submit` used to fail one prerequisite at a time
(no npa source, then no SkyPilot CLI, then a placeholder bucket), each as a
separate run, and there was no command to produce the npa source copy at all.
"""

from __future__ import annotations

from contextlib import nullcontext
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
import typer
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.cli.workbench import workflow as workflow_cli
from npa.orchestration.npa_workflow.skypilot_render import ImagePullRequirements
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.robotwin_preflight import (
    CUSTOMER_ENTITLEMENT_ENV as ROBOTWIN_ENTITLEMENT_ENV,
    CUSTOMER_TERMS,
    CUSTOMER_USE_SCOPE,
    MATERIALIZED_KUBECONFIG_ENV,
    MATERIALIZED_SKYPILOT_CONFIG_ENV,
    PUBLIC_CONTEXT_ENV as ROBOTWIN_CONTEXT_ENV,
    RUNTIME_LOCK_SHA256,
    TRANSPORT_CONTEXT_ENV as ROBOTWIN_TRANSPORT_ENV,
)

runner = CliRunner()
_REAL_EXECUTION_TARGET_PREFLIGHT = workflow_cli._execution_target_preflight

_OPERATOR_REGISTRY = "registry.example.invalid/operator/workbench"

SPEC = (
    Path(__file__).resolve().parents[3]
    / "workflows"
    / "testing"
    / "physical-ai-data-factory.yaml"
)
COSMOS3_SPEC = (
    Path(__file__).resolve().parents[3] / "workflows" / "main" / "paidf-cosmos3.yaml"
)
NVIDIA_VDA_SPEC = (
    Path(__file__).resolve().parents[3]
    / "workflows"
    / "testing"
    / "nvidia-paidf-vda-cosmos-transfer25.yaml"
)
SIM2REAL_SPEC = (
    Path(__file__).resolve().parents[3] / "workflows" / "main" / "sim2real.yaml"
)
SONIC_SPEC = (
    Path(__file__).resolve().parents[3]
    / "workflows"
    / "testing"
    / "sonic-export-eval.yaml"
)
ROBOTWIN_SPEC = (
    Path(__file__).resolve().parents[3] / "workflows" / "testing" / "byof-robotwin.yaml"
)


def test_nvidia_vda_conditioning_policy_is_versioned_by_the_committed_spec() -> None:
    from npa.cli.workbench.workflow import _paidf_conditioning_policy
    from npa.orchestration.npa_workflow.run_state import (
        NVIDIA_PAIDF_VDA_WORKFLOW_NAME,
    )

    assert (
        _paidf_conditioning_policy(NVIDIA_PAIDF_VDA_WORKFLOW_NAME, {})
        == "source-fidelity-v2"
    )
    assert (
        _paidf_conditioning_policy(
            NVIDIA_PAIDF_VDA_WORKFLOW_NAME,
            {"input_conditioning_policy": "source-fidelity-v3"},
        )
        == "source-fidelity-v3"
    )
    assert (
        _paidf_conditioning_policy(
            "physical-ai-data-factory",
            {"input_conditioning_policy": "source-fidelity-v3"},
        )
        == ""
    )


@pytest.fixture(autouse=True)
def _no_ambient_src(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests isolate prerequisite/image/provisioning ordering. Exact
    # provider scope and prefix probes have independent CLI contract coverage.
    monkeypatch.setattr(
        workflow_cli, "_execution_target_preflight", lambda *args, **kwargs: (None, {})
    )
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)


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
            "--registry",
            _OPERATOR_REGISTRY,
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
            "--registry",
            _OPERATOR_REGISTRY,
            *args,
        ],
    )


def _submit_robotwin(*args: str, run_id: str = "robotwin-public-launcher"):
    return runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(ROBOTWIN_SPEC),
            "--run-id",
            run_id,
            "--no-deploy-if-absent",
            *args,
        ],
    )


def _submit_nvidia_vda(*args: str):
    return runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(NVIDIA_VDA_SPEC),
            "--run-id",
            "nvidia-vda-preflight-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--no-deploy-if-absent",
            "--registry",
            _OPERATOR_REGISTRY,
            *args,
        ],
    )


def _install_robotwin_submit_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    assertion_decision: str = "accepted",
    install_authenticated_boundary: bool = True,
) -> tuple[dict[str, object], Path]:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.robotwin_preflight.RUNTIME_LOCK_STATUS",
        "complete",
    )
    kubeconfig = tmp_path / "kubeconfig.yaml"
    kubeconfig.write_text(
        "apiVersion: v1\nkind: Config\n"
        "current-context: robotwin-context\n"
        "clusters: [{name: robotwin-cluster, cluster: {server: "
        "https://cluster.example.invalid, certificate-authority-data: Y2E=}}]\n"
        "contexts: [{name: robotwin-context, context: {cluster: "
        "robotwin-cluster, user: robotwin-user}}]\n"
        "users: [{name: robotwin-user, user: {token: portable-test-token}}]\n",
        encoding="utf-8",
    )
    skypilot = tmp_path / "skypilot.yaml"
    skypilot.write_text(
        "kubernetes:\n  allowed_contexts: [robotwin-context]\n",
        encoding="utf-8",
    )
    for path in (kubeconfig, skypilot):
        path.chmod(0o600)
    materialized_dir = tmp_path / "materialized-config"
    materialized_dir.mkdir(mode=0o700)
    materialized_kubeconfig = materialized_dir / "kubeconfig.yaml"
    materialized_skypilot = materialized_dir / "skypilot.yaml"
    materialized_kubeconfig.write_bytes(kubeconfig.read_bytes())
    materialized_skypilot.write_bytes(skypilot.read_bytes())
    materialized_kubeconfig.chmod(0o600)
    materialized_skypilot.chmod(0o600)
    payload: dict[str, object] = {
        "solution": "robotwin",
        "ownership_provenance": "manager-issued",
        "customer_scope_id": "robotwin-customer-canary",
        "workflow_sha256": "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3",
        "source_revision": "96c1feab536306b50c26af200044fcdf126e8904",
        "curobo_revision": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
        "asset_revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad",
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "bootstrap_image": "registry.example/robotwin-private/npa-robotwin@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "reservation": {
            "policy": "STRICT",
            "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
            "count": 1,
        },
        "project": "robotwin-private-project-canary",
        "nebius_profile": "robotwin-private-profile-canary",
        "kubeconfig": str(kubeconfig),
        "kubernetes_context": "robotwin-context",
        "skypilot_config_path": str(skypilot),
        "bucket": "robotwin-private-bucket-canary",
        "output_root": "s3://robotwin-private-bucket-canary/output",
        "run_id": "robotwin-private-run-canary",
    }
    context = tmp_path / "runtime-context.json"
    context.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    context.chmod(0o600)
    monkeypatch.setenv(ROBOTWIN_CONTEXT_ENV, str(context))
    monkeypatch.setenv(MATERIALIZED_KUBECONFIG_ENV, str(materialized_kubeconfig))
    monkeypatch.setenv(MATERIALIZED_SKYPILOT_CONFIG_ENV, str(materialized_skypilot))
    unsigned_entitlement = tmp_path / "unsigned-customer-entitlement.json"
    unsigned_entitlement.write_text(
        json.dumps(
            {
                "provenance": "customer-issued",
                "customer_scope_id": payload["customer_scope_id"],
                "run_id": payload["run_id"],
                "runtime_manifest_sha256": payload["runtime_lock_sha256"],
                "expires_at": "2099-01-01T00:00:00Z",
                "decision": assertion_decision,
                "intended_activity": CUSTOMER_USE_SCOPE,
                "terms": list(CUSTOMER_TERMS),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    unsigned_entitlement.chmod(0o600)
    monkeypatch.delenv(ROBOTWIN_ENTITLEMENT_ENV, raising=False)
    if install_authenticated_boundary:
        from npa.orchestration.npa_workflow import robotwin_preflight

        real_prepare = robotwin_preflight.prepare_live_submit
        assertion = SimpleNamespace(
            issuer="https://customer-auth.example.invalid",
            customer_scope_id=payload["customer_scope_id"],
            run_id=payload["run_id"],
            runtime_manifest_sha256=payload["runtime_lock_sha256"],
            issued_at="2026-01-01T00:00:00Z",
            expires_at="2099-01-01T00:00:00Z",
            decision=assertion_decision,
            intended_activity=CUSTOMER_USE_SCOPE,
            terms=list(CUSTOMER_TERMS),
            assertion_id="assertion-cli-canary-0001",
            nonce="nonce-cli-canary-00000001",
        )
        boundary = SimpleNamespace(
            trusted_issuer="https://customer-auth.example.invalid",
            consume_once=lambda _request: assertion,
        )

        def prepare_with_authenticated_boundary(*args, **kwargs):
            kwargs["customer_authorization_boundary"] = boundary
            return real_prepare(*args, **kwargs)

        monkeypatch.setattr(
            robotwin_preflight,
            "prepare_live_submit",
            prepare_with_authenticated_boundary,
        )
    else:
        monkeypatch.setattr(
            workflow_cli,
            "_robotwin_unsigned_entitlement_for_test",
            unsigned_entitlement,
            raising=False,
        )
    return payload, context


@pytest.mark.parametrize("malformed", [False, True])
def test_robotwin_submit_refuses_before_every_external_boundary_even_when_skipped(
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    tmp_path: Path,
    malformed: bool,
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.robotwin_preflight.RUNTIME_LOCK_STATUS",
        "complete",
    )
    monkeypatch.delenv(ROBOTWIN_CONTEXT_ENV, raising=False)
    args = [
        "--secret-env",
        ROBOTWIN_CONTEXT_ENV,
        "--skip-preflight",
    ]
    if malformed:
        context = tmp_path / "robotwin-context.json"
        context.write_text("{", encoding="utf-8")
        context.chmod(0o600)
        monkeypatch.setenv(ROBOTWIN_CONTEXT_ENV, str(context))
    boundaries = [
        mocker.patch("npa.orchestration.npa_workflow.first_run_state.prepare_run"),
        mocker.patch(
            "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials"
        ),
        mocker.patch("npa.cli.workbench.workflow._resolve_submit_registry"),
        mocker.patch("npa.cli.workbench.workflow._preflight_submit_images"),
        mocker.patch("npa.cli.workbench.workflow._execution_target_preflight"),
        mocker.patch("npa.cli.workbench.workflow._preflight_submit_gang_capacity"),
        mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit"),
        mocker.patch("npa.orchestration.npa_workflow.deploy.ensure_infra_present"),
        mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow"),
    ]

    result = _submit_robotwin(*args)

    assert result.exit_code == 1, result.output
    expected = "context-invalid-json" if malformed else "context-missing"
    assert expected in result.output
    for boundary in boundaries:
        boundary.assert_not_called()


@pytest.mark.parametrize(
    ("missing", "category"),
    [
        (True, "needs_customer_acceptance"),
        (False, "customer-authorization-unsigned-local-file"),
    ],
)
def test_robotwin_customer_entitlement_refuses_before_external_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    tmp_path: Path,
    missing: bool,
    category: str,
) -> None:
    from npa.orchestration.npa_workflow import robotwin_preflight

    _install_robotwin_submit_context(
        monkeypatch,
        tmp_path,
        assertion_decision="declined",
        install_authenticated_boundary=False,
    )
    source_fingerprint = "a" * 64
    source_uri = f"s3://fixture-source-bucket/npa-src/{source_fingerprint}"
    source_proof_calls: list[tuple[str, str]] = []

    def prove_source_bytes(uri: str, fingerprint: str) -> None:
        source_proof_calls.append((uri, fingerprint))

    monkeypatch.setenv("NPA_SRC_S3_URI", source_uri)
    monkeypatch.setattr(
        workflow_cli, "_local_source_fingerprint", lambda: source_fingerprint
    )
    monkeypatch.setattr(
        robotwin_preflight,
        "_require_verified_control_plane_source_bytes",
        prove_source_bytes,
    )
    secret_args: tuple[str, ...] = ()
    if not missing:
        unsigned = workflow_cli._robotwin_unsigned_entitlement_for_test
        monkeypatch.setenv(ROBOTWIN_ENTITLEMENT_ENV, str(unsigned))
        secret_args = ("--secret-env", ROBOTWIN_ENTITLEMENT_ENV)
    boundaries = [
        mocker.patch("npa.orchestration.npa_workflow.first_run_state.prepare_run"),
        mocker.patch(
            "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials"
        ),
        mocker.patch("npa.cli.workbench.workflow._resolve_submit_registry"),
        mocker.patch("npa.cli.workbench.workflow._preflight_submit_images"),
        mocker.patch("npa.cli.workbench.workflow._execution_target_preflight"),
        mocker.patch("npa.cli.workbench.workflow._preflight_submit_gang_capacity"),
        mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit"),
        mocker.patch("npa.orchestration.npa_workflow.deploy.ensure_infra_present"),
        mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow"),
    ]

    result = _submit_robotwin(
        "--secret-env",
        ROBOTWIN_CONTEXT_ENV,
        *secret_args,
        "--skip-preflight",
    )

    assert result.exit_code == 1
    assert category in result.output
    assert CUSTOMER_TERMS[0]["url"] in result.output
    expected_source_proof_calls = [(source_uri, source_fingerprint)] if missing else []
    assert source_proof_calls == expected_source_proof_calls
    for boundary in boundaries:
        boundary.assert_not_called()


def test_relabelled_robotwin_workflow_refuses_before_external_boundaries(
    mocker, tmp_path: Path
) -> None:
    document = yaml.safe_load(ROBOTWIN_SPEC.read_text(encoding="utf-8"))
    document["metadata"]["name"] = "generic-workflow"
    document["config"].update(
        {
            "solution_name": "generic-solution",
            "runtime_context_env": "GENERIC_RUNTIME_CONTEXT",
            "repo_ref": "main",
        }
    )
    relabelled = tmp_path / "generic-workflow.yaml"
    relabelled.write_text(yaml.safe_dump(document), encoding="utf-8")
    boundaries = [
        mocker.patch("npa.orchestration.npa_workflow.first_run_state.prepare_run"),
        mocker.patch(
            "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials"
        ),
        mocker.patch("npa.cli.workbench.workflow._resolve_submit_registry"),
        mocker.patch("npa.cli.workbench.workflow._preflight_submit_images"),
        mocker.patch("npa.cli.workbench.workflow._execution_target_preflight"),
        mocker.patch("npa.cli.workbench.workflow._preflight_submit_gang_capacity"),
        mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit"),
        mocker.patch("npa.orchestration.npa_workflow.deploy.ensure_infra_present"),
        mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow"),
    ]

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(relabelled),
            "--run-id",
            "generic-public-launcher",
            "--no-deploy-if-absent",
            "--skip-preflight",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "workflow-config" in result.output
    for boundary in boundaries:
        boundary.assert_not_called()


def test_robotwin_plan_only_is_context_free_and_publicly_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ROBOTWIN_CONTEXT_ENV, "private-context-path-canary")
    monkeypatch.setenv(ROBOTWIN_ENTITLEMENT_ENV, "private-entitlement-path-canary")
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.robotwin_preflight.read_owner_context",
        lambda *_args, **_kwargs: pytest.fail("plan-only read private context"),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.robotwin_preflight.read_customer_entitlement",
        lambda *_args, **_kwargs: pytest.fail("plan-only read customer entitlement"),
    )

    result = _submit_robotwin("--plan-only", "--output-format", "json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    serialized = json.dumps(payload, sort_keys=True)
    assert ROBOTWIN_CONTEXT_ENV in serialized
    assert ROBOTWIN_TRANSPORT_ENV not in serialized
    assert "private-context-path-canary" not in serialized
    assert "private-entitlement-path-canary" not in serialized
    assert "example-bucket" in serialized


def test_non_robotwin_live_submit_never_enters_robotwin_preflight(
    monkeypatch: pytest.MonkeyPatch,
    mocker,
) -> None:
    special = mocker.patch(
        "npa.orchestration.npa_workflow.robotwin_preflight.prepare_live_submit"
    )

    result = _submit("--skip-preflight")

    assert result.exit_code != 0
    assert (
        "HF_TOKEN is required to verify the exact gated Cosmos Transfer checkpoint "
        "before provisioning or GPU work" in result.output
    )
    special.assert_not_called()


@pytest.mark.parametrize("output_format", ["json", "text"])
def test_robotwin_normal_submit_redacts_private_values_in_all_output_modes(
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    tmp_path: Path,
    output_format: str,
) -> None:
    from npa.orchestration.npa_workflow import robotwin_preflight
    from npa.orchestration.skypilot.workflow import WorkflowResult

    monkeypatch.setattr(
        robotwin_preflight,
        "_require_verified_control_plane_source_bytes",
        lambda _uri, _fingerprint: None,
    )
    private, context_path = _install_robotwin_submit_context(monkeypatch, tmp_path)
    _mock_sky_bin_ok(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path / "npa-config"))
    summary_uri = f"{private['output_root']}/{private['run_id']}/npa_byof_summary.json"
    credentials = SimpleNamespace(
        endpoint_url="https://storage.eu-north1.nebius.cloud",
        secret_values={
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        },
        missing=(),
        access_key_id="test-access-key",
        secret_access_key="test-secret-key",
    )
    captured: dict[str, object] = {}
    receipts: list[dict[str, object]] = []

    mocker.patch(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        return_value=credentials,
    )
    source_uri = "s3://public-source-bucket/npa-src/npa/" + "a" * 64
    monkeypatch.setenv("NPA_SRC_S3_URI", source_uri)
    saved_source_resolver = mocker.patch(
        "npa.cli.workbench.workflow._resolve_submit_src_s3_uri_with_origin",
        return_value=(source_uri, "environment"),
    )
    mocker.patch(
        "npa.cli.workbench.workflow._local_source_fingerprint",
        return_value="a" * 64,
    )
    mocker.patch("npa.cli.workbench.workflow._submit_prerequisites", return_value=[])
    mocker.patch("npa.cli.workbench.workflow._preflight_submit_images", return_value={})
    mocker.patch("npa.cli.workbench.workflow._verify_submit_controller_owner")
    mocker.patch("npa.execution_preflight.verify_execution_scope", return_value={})
    mocker.patch("npa.provisioning_journal.current_operation", return_value=None)
    operation_prepare = mocker.patch(
        "npa.provisioning_journal.ProvisioningOperation.prepare"
    )
    mocker.patch(
        "npa.orchestration.npa_workflow.submission_state.submission_lock",
        return_value=nullcontext(),
    )
    mocker.patch(
        "npa.orchestration.npa_workflow.submission_state.update_submission_state",
        side_effect=lambda _project, _run_id, value, **_kwargs: receipts.append(value),
    )
    mocker.patch("npa.orchestration.npa_workflow.src_staging._storage_client")
    mocker.patch("npa.orchestration.npa_workflow.src_staging.verify_staged_source")

    def execution_target(*_args, **kwargs):
        captured["authorized_output_uri"] = kwargs["authorized_output_uri"]
        return object(), {"execution_readiness": "pass"}

    def submit_workflow(*args, **kwargs):
        captured["submit_args"] = args
        captured["submit_kwargs"] = kwargs
        captured["rendered"] = Path(args[0]).read_text(encoding="utf-8")
        return WorkflowResult(
            status="SUBMITTED",
            job_id="public-job-canary",
            stdout=f"downstream mentioned {summary_uri}",
        )

    mocker.patch(
        "npa.cli.workbench.workflow._execution_target_preflight",
        side_effect=execution_target,
    )
    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=submit_workflow,
    )
    stage = mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit")

    result = _submit_robotwin(
        "--secret-env",
        ROBOTWIN_CONTEXT_ENV,
        "--secret-env",
        "AWS_ACCESS_KEY_ID",
        "--secret-env",
        "AWS_SECRET_ACCESS_KEY",
        "--skip-preflight",
        "--isolated-config-dir",
        str(tmp_path / "sky-state"),
        "--output-format",
        output_format,
        run_id=str(private["run_id"]),
    )

    assert result.exit_code == 0, result.output
    assert captured["authorized_output_uri"] == summary_uri
    submit_kwargs = captured["submit_kwargs"]
    assert submit_kwargs["sky_bin"] == "/usr/bin/sky"
    assert ROBOTWIN_CONTEXT_ENV not in submit_kwargs["secret_envs"]
    assert ROBOTWIN_ENTITLEMENT_ENV not in submit_kwargs["secret_envs"]
    assert ROBOTWIN_TRANSPORT_ENV in submit_kwargs["secret_envs"]
    assert ROBOTWIN_TRANSPORT_ENV in submit_kwargs["extra_env"]
    assert set(submit_kwargs["secret_envs"]) == {
        ROBOTWIN_TRANSPORT_ENV,
        "NPA_SRC_S3_URI",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "NEBIUS_S3_ENDPOINT",
    }
    assert submit_kwargs["infra"] == "k8s/robotwin-context"
    assert submit_kwargs["robotwin_submit_context"] is not None
    assert submit_kwargs["execution_preflight_report"] == {
        "execution_readiness": "pass"
    }
    assert str(context_path) not in json.dumps(submit_kwargs["secret_envs"])
    rendered = str(captured["rendered"])
    assert ROBOTWIN_CONTEXT_ENV in rendered
    assert ROBOTWIN_TRANSPORT_ENV not in rendered
    assert source_uri not in rendered
    rendered_environment = list(yaml.safe_load_all(rendered))[1]["envs"]
    assert rendered_environment["NPA_SRC_S3_URI"] == "${NPA_SRC_S3_URI}"
    assert rendered_environment["AWS_ENDPOINT_URL"] == "${AWS_ENDPOINT_URL}"
    for field in (
        "project",
        "nebius_profile",
        "kubeconfig",
        "kubernetes_context",
        "skypilot_config_path",
        "bootstrap_image",
        "bucket",
        "output_root",
    ):
        value = str(private[field])
        assert value not in rendered
        assert value not in result.output
    assert str(private["run_id"]) not in result.output
    assert summary_uri not in result.output
    if output_format == "json":
        assert "<redacted>" in json.loads(result.stdout)["stdout"]
    else:
        assert str(private["run_id"]) not in result.stdout
        assert str(private["run_id"]) not in result.stderr
        assert "run_id: <redacted>" in result.stdout
        assert "Reserved fresh run <redacted>" in result.stderr
    assert receipts == []
    operation_prepare.assert_not_called()
    saved_source_resolver.assert_not_called()
    assert source_uri not in result.output
    stage.assert_not_called()


def test_robotwin_missing_control_plane_source_refuses_without_state_or_staging(
    monkeypatch: pytest.MonkeyPatch, mocker, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    _install_robotwin_submit_context(monkeypatch, tmp_path)
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path / "npa-config"))
    mocker.patch(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        return_value=SimpleNamespace(
            endpoint_url="https://storage.eu-north1.nebius.cloud",
            secret_values={},
            missing=(),
            access_key_id="test-access-key",
            secret_access_key="test-secret-key",
        ),
    )
    mocker.patch(
        "npa.cli.workbench.workflow._resolve_submit_src_s3_uri_with_origin",
        return_value=("", "default"),
    )
    mocker.patch(
        "npa.cli.workbench.workflow._local_source_fingerprint",
        return_value="a" * 64,
    )
    stage = mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit")
    state = mocker.patch(
        "npa.orchestration.npa_workflow.submission_state.update_submission_state"
    )
    launch = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")
    operation = mocker.patch("npa.provisioning_journal.ProvisioningOperation.prepare")

    result = _submit_robotwin(
        "--secret-env",
        ROBOTWIN_CONTEXT_ENV,
        "--skip-preflight",
        "--output-format",
        "json",
    )

    assert result.exit_code != 0
    assert "control-plane-source-explicit-uri-required" in result.output
    stage.assert_not_called()
    state.assert_not_called()
    launch.assert_not_called()
    operation.assert_not_called()


def test_robotwin_authorized_output_stays_runtime_only_not_in_submission_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from npa.orchestration.npa_workflow import build_plan, load_spec

    actual = (
        "s3://private-bucket-canary/output/robotwin-run-canary/npa_byof_summary.json"
    )
    captured: dict[str, object] = {}

    def resolve_execution_target(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_uris=tuple(kwargs["output_uris"]))

    monkeypatch.setattr(
        "npa.execution_preflight.resolve_execution_target", resolve_execution_target
    )
    monkeypatch.setattr(
        "npa.execution_preflight.verify_execution_target",
        lambda target, **_kwargs: {"verified": target.output_uris},
    )
    spec = load_spec(ROBOTWIN_SPEC)

    target, _report = _REAL_EXECUTION_TARGET_PREFLIGHT(
        spec,
        project="private-project-canary",
        context="private-context-canary",
        region="",
        run_id="robotwin-public-launcher",
        assume_decision="",
        credentials=SimpleNamespace(),
        authorized_output_uri=actual,
        verify_cluster=False,
    )
    prepared = SimpleNamespace(
        spec=spec,
        plan=SimpleNamespace(
            steps=build_plan(spec, run_id="robotwin-public-launcher").steps
        ),
    )
    receipt = workflow_cli._npa_submission_receipt(prepared, "robotwin-public-launcher")

    assert target.output_uris == (actual,)
    assert captured["output_uris"] == [actual]
    assert "authorization" not in receipt
    receipt_outputs = [
        output["uri"] for step in receipt["steps"] for output in step.get("outputs", [])
    ]
    assert receipt_outputs == [
        "s3://example-bucket/oss-solutions/robotwin/"
        "robotwin-public-launcher/npa_byof_summary.json"
    ]
    assert actual not in json.dumps(receipt, sort_keys=True)
    assert actual not in json.dumps(
        build_plan(spec, run_id="robotwin-public-launcher").to_dict()
    )


def test_robotwin_source_staging_is_not_a_workload_output_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from npa.orchestration.npa_workflow import load_spec

    summary = "s3://manager-bucket/output/robotwin-run/npa_byof_summary.json"
    source = "s3://control-bucket/npa-source/exact-head"
    resolved: list[dict[str, object]] = []

    def resolve_execution_target(**kwargs):
        resolved.append(kwargs)
        return SimpleNamespace(**kwargs)

    def verify_execution_target(target, **_kwargs):
        return {
            "presence": "pass",
            "access": "pass",
            "destination_count": len(target.output_uris),
        }

    monkeypatch.setattr(
        "npa.execution_preflight.resolve_execution_target", resolve_execution_target
    )
    monkeypatch.setattr(
        "npa.execution_preflight.verify_execution_target", verify_execution_target
    )

    target, report = _REAL_EXECUTION_TARGET_PREFLIGHT(
        load_spec(ROBOTWIN_SPEC),
        project="manager-project",
        context="manager-context",
        region="",
        run_id="robotwin-public-launcher",
        assume_decision="",
        credentials=SimpleNamespace(),
        source_uri=source,
        authorized_output_uri=summary,
        verify_cluster=False,
    )

    assert target.output_uris == [summary]
    assert report["destination_count"] == 1
    assert resolved[0]["output_uris"] == [summary]
    assert resolved[1]["output_uris"] == [source + "/"]
    assert resolved[1]["provenance"] == {"outputs": "control-plane-source-staging"}
    assert report["control_plane_source_staging"] == {
        "presence": "pass",
        "access": "pass",
        "destination_count": 1,
    }


def _resumed_preflight_options(source: str, recorded: str) -> dict:
    return dict(
        project="unit",
        context="unit",
        region="",
        run_id="resume-run",
        assume_decision="",
        credentials=SimpleNamespace(),
        verify_cluster=False,
        recorded_store=SimpleNamespace(run_prefix_uri=recorded),
        source_uri=source,
    )


@pytest.mark.parametrize("source_denied", [False, True])
def test_resume_preflight_checks_control_and_source_destinations_separately(
    monkeypatch: pytest.MonkeyPatch, source_denied: bool
) -> None:
    from npa.orchestration.npa_workflow import load_spec

    source = "s3://control-bucket/source/frozen"
    recorded = "s3://control-bucket/recorded-run"
    verified = []
    monkeypatch.setattr(
        "npa.execution_preflight.resolve_execution_target",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def verify(target, **_kwargs):
        verified.append(target)
        if source_denied and source + "/" in target.output_uris:
            raise ValueError("source staging denied")
        return {"presence": "pass", "access": "pass"}

    monkeypatch.setattr("npa.execution_preflight.verify_execution_target", verify)
    kwargs = _resumed_preflight_options(source, recorded)
    if source_denied:
        with pytest.raises(ValueError, match="source staging denied"):
            _REAL_EXECUTION_TARGET_PREFLIGHT(load_spec(SPEC), **kwargs)
    else:
        _REAL_EXECUTION_TARGET_PREFLIGHT(load_spec(SPEC), **kwargs)
    assert recorded + "/" in verified[0].output_uris
    assert source + "/" not in verified[0].output_uris
    assert verified[1].output_uris == [source + "/"]
    assert verified[1].provenance == {"outputs": "control-plane-source-staging"}


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


def test_fail_discards_the_handled_exception_graph(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        workflow_cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )

    try:
        raise ValueError("private-context-document-canary")
    except ValueError:
        with pytest.raises(typer.Exit) as exc_info:
            workflow_cli._fail("sanitized refusal")

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "private-context-document-canary" not in output.getvalue()


def test_robotwin_private_submit_values_are_redacted_from_errors_and_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = "s3://private-bucket-canary/private-output"
    transport = "private-transport-value-canary"
    token = workflow_cli._SUBMIT_PRIVATE_REDACTIONS.set((private, transport))
    output = StringIO()
    monkeypatch.setattr(
        workflow_cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    try:
        with pytest.raises(typer.Exit):
            workflow_cli._fail(f"target {private} carried {transport}")
        redacted = workflow_cli._redact_submit_private_values(
            {"error": private, "nested": [f"prefix:{transport}"]}
        )
    finally:
        workflow_cli._SUBMIT_PRIVATE_REDACTIONS.reset(token)

    assert private not in output.getvalue()
    assert transport not in output.getvalue()
    assert output.getvalue() == "Error: target <redacted> carried <redacted>\n"
    assert redacted == {
        "error": "<redacted>",
        "nested": ["prefix:<redacted>"],
    }


def test_fail_structurally_redacts_raw_multiline_exception(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        workflow_cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    message = (
        "provider rejected HF_TOKEN=hf_synthetic_boundary_token\n"
        "retry: npa workbench workflow status synthetic-run\n"
        "details: https://synthetic-user:synthetic-password@api.example.invalid/"
        "path?X-Amz-Signature=synthetic-query"
    )

    with pytest.raises(typer.Exit) as exc_info:
        workflow_cli._fail(message)

    rendered = output.getvalue()
    assert exc_info.value.exit_code == 1
    assert rendered.count("\n") == message.count("\n") + 1
    assert "retry: npa workbench workflow status synthetic-run" in rendered
    for secret in (
        "hf_synthetic_boundary_token",
        "synthetic-user",
        "synthetic-password",
        "X-Amz-Signature",
        "synthetic-query",
    ):
        assert secret not in rendered


def test_submit_lists_every_missing_prerequisite_at_once(tmp_path: Path) -> None:
    result = _submit("--sky-bin", str(tmp_path / "missing-sky"))

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
    _mock_sky_bin_ok(monkeypatch)
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
    mocker.patch("npa.cli.workbench.workflow._preflight_submit_images", return_value={})
    storage = mocker.patch("npa.clients.storage_validation.probe_storage_write")
    prepare_input = mocker.patch("npa.workflows.data_factory_input.prepare_paidf_input")
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


@pytest.mark.parametrize("control", ["edge", "vis", "seg"])
def test_paidf_existing_target_orders_placement_exact_access_then_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    control: str,
) -> None:
    from npa.orchestration.npa_workflow.spec import load_spec

    _mock_sky_bin_ok(monkeypatch)

    # The selected Transfer tool routes through cosmos2, so catalog expansion
    # must not broaden PAIDF's deliberately narrow submit-time access fence.
    assert {
        item.repo
        for item in workflow_cli._workflow_access_requirements(load_spec(SPEC))
    } == {
        "nvidia/Cosmos-Transfer2.5-2B",
        "nvidia/Cosmos-Guardrail1",
        "nvidia/Cosmos-Predict2.5-2B",
    }
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "access.json"))
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
            "--var",
            f"augment_control={control}",
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
    assert events == ["placement", f"exact:{control}", "image"]


def test_checkpoint_access_failure_redacts_resolved_opaque_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workbench.cosmos.checkpoint_access import CosmosCheckpointAccessError

    opaque_token = "synthetic-opaque-checkpoint-credential"
    _mock_sky_bin_ok(monkeypatch)
    monkeypatch.setenv("NPA_ACCESS_APPROVAL_STATE_PATH", str(tmp_path / "access.json"))
    for name, value in (
        ("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-token-factory-key"),
        ("AWS_ACCESS_KEY_ID", "synthetic-access-key"),
        ("AWS_SECRET_ACCESS_KEY", "synthetic-secret-key"),
        ("HF_TOKEN", opaque_token),
    ):
        monkeypatch.setenv(name, value)
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
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._paidf_kubernetes_prerequisites_for_submit",
        lambda _context: [],
    )

    def denied(*, modality: str, token: str):
        assert modality == "edge"
        assert token == opaque_token
        raise CosmosCheckpointAccessError(f"provider rejected token {token}")

    monkeypatch.setattr(
        "npa.workbench.cosmos.checkpoint_access.preflight_control_checkpoint_access",
        denied,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._preflight_submit_images",
        lambda *_args, **_kwargs: pytest.fail(
            "image preflight reached after checkpoint access failure"
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--run-id",
            "paidf-redacted-access-failure",
            "--no-deploy-if-absent",
            "--var",
            "bucket=real-bucket",
            "--var",
            "augment_control=edge",
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
    assert "provider rejected token <redacted>" in result.output
    assert opaque_token not in result.output


@pytest.mark.parametrize(
    ("count_name", "count"),
    [("rollout_count", "513"), ("validation_count", "65"), ("gold_count", "65")],
)
def test_sim2real_submit_rejects_oversized_sealed_split_before_images_or_launch(
    monkeypatch: pytest.MonkeyPatch, mocker, count_name: str, count: str
) -> None:
    from types import SimpleNamespace

    secret_names = (
        "NEBIUS_TOKEN_FACTORY_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
    )
    for name in secret_names:
        monkeypatch.setenv(name, "redacted")
    mocker.patch("npa.cli.workbench.workflow._submit_prerequisites", return_value=[])
    mocker.patch(
        "npa.orchestration.npa_workflow.sim2real_preflight.kubernetes_prerequisites",
        return_value=[],
    )
    mocker.patch(
        "npa.clients.huggingface.validate_hf_access",
        return_value=SimpleNamespace(ok=True),
    )
    mocker.patch(
        "npa.clients.token_factory.validate_model_access",
        return_value=SimpleNamespace(ok=True),
    )
    images = mocker.patch("npa.cli.workbench.workflow._preflight_submit_images")
    launch = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")
    runtime = mocker.patch("npa.cli.workbench.workflow._run_npa_workflow_runtime")
    stage = mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit")
    config = {
        "bucket": "test-bucket",
        "source_sha": "a" * 40,
        "isaac_cache_pvc": "test-isaac-cache",
        "env_count": "640",
        "train_fraction": "0.8",
        "rollout_count": "64",
        "validation_count": "64",
        "gold_count": "64",
        **{
            key: "ghcr.io/example/test@sha256:" + "a" * 64
            for key in (
                "controller_image",
                "transfer_image",
                "envgen_image",
                "isaac_image",
                "viewer_image",
            )
        },
        count_name: count,
    }
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SIM2REAL_SPEC),
            "--run-id",
            "sim2real-split-preflight",
            "--no-deploy-if-absent",
            *[
                arg
                for key, value in config.items()
                for arg in ("--var", f"{key}={value}")
            ],
            *[arg for name in secret_names for arg in ("--secret-env", name)],
        ],
    )

    assert result.exit_code == 1, result.output
    assert "sealed train/validation/gold" in result.output
    assert "available=512/64/64" in result.output
    images.assert_not_called()
    launch.assert_not_called()
    runtime.assert_not_called()
    stage.assert_not_called()


def test_sim2real_submit_propagates_explicit_kubernetes_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from subprocess import CompletedProcess

    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return CompletedProcess(args, 1, stdout="", stderr="NotFound")

    monkeypatch.setenv("KUBECONFIG", "/tmp/sim2real-kubeconfig")
    monkeypatch.setenv("NPA_SIM2REAL_K8S_NAMESPACE", "sim2real-benchmark")
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
            "--var",
            "isaac_cache_pvc=npa-isaac-cache",
        ],
    )

    assert result.exit_code == 1
    assert calls
    assert all(call[1]["context"] == "sim2real-review" for call in calls)
    assert all(call[1]["kubeconfig"] == "/tmp/sim2real-kubeconfig" for call in calls)
    namespaced_calls = [call[0] for call in calls if call[0][:2] == ["get", "pvc"]]
    assert namespaced_calls
    assert all(
        args[args.index("-n") + 1] == "sim2real-benchmark" for args in namespaced_calls
    )


def test_submit_preflight_clears_as_prerequisites_are_met(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each satisfied prerequisite drops out of the report."""
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(tmp_path / "missing-sky"))
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


def test_plan_only_reports_quarantined_default_as_cli_error() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--run-id",
            "quarantine-cli-contract",
            "--assume-decision",
            "promote_checkpoint",
            "--no-deploy-if-absent",
            "--plan-only",
            "--var",
            "bucket=real-bucket",
        ],
    )

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "no consumable public release" in result.output
    assert "operator-controlled registry/image" in result.output
    assert not isinstance(result.exception, ValueError)


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


def test_nvidia_vda_plan_stages_input_beneath_its_own_run_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")

    result = _submit_nvidia_vda(
        "--plan-only", "--var", "bucket=real-bucket", "--output-format", "json"
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["plan"]["steps"][0]["state"] == "record-upstream"
    serialized_plan = json.dumps(payload["plan"], sort_keys=True)
    assert (
        "s3://real-bucket/nvidia-paidf-vda-cosmos-transfer25/"
        "nvidia-vda-preflight-demo/input/"
    ) in serialized_plan


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


@pytest.fixture
def paidf_input_submit_ready(monkeypatch, mocker):
    _mock_sky_bin_ok(monkeypatch)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://artifacts/npa-src/npa")
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf-token")
    monkeypatch.setattr(workflow_cli, "_adopt_npa_kubeconfig", lambda _: True)
    mocker.patch.object(workflow_cli, "_verify_submit_controller_owner")
    mocker.patch.object(workflow_cli, "_preflight_submit_images", return_value={})
    mocker.patch.object(workflow_cli, "_enforce_workflow_access")
    mocker.patch(
        "npa.workbench.cosmos.checkpoint_access.preflight_control_checkpoint_access"
    )


def test_paidf_listing_diagnosis_reaches_submit_without_private_error_text(
    monkeypatch, mocker, paidf_input_submit_ready
):
    from botocore.exceptions import ClientError
    from npa.workflows import data_factory_input as dfi

    monkeypatch.setattr(dfi.shutil, "which", lambda name: f"/test-bin/{name}")
    storage = mocker.Mock()
    storage.s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey"}}, "GetObject"
    )
    storage.s3.get_paginator.return_value.paginate.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "private/capture.mp4"}},
        "ListObjectsV2",
    )
    mocker.patch(
        "npa.clients.storage.StorageClient.from_environment", return_value=storage
    )
    launch = mocker.patch.object(workflow_cli, "_run_npa_workflow_runtime")

    result = _submit("--skip-preflight", "--var", "bucket=artifacts")

    assert result.exit_code == 1, result.output
    assert "ClientError (AccessDenied)" in result.output
    assert "private/capture.mp4" not in result.output
    storage.s3.get_paginator.return_value.paginate.assert_called_once()
    storage.download_path.assert_not_called()
    launch.assert_not_called()


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


def test_skip_preflight_bypasses_static_prerequisites(mocker) -> None:
    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=AssertionError("submit reached"),
    )

    result = _submit("--skip-preflight")

    # Convenience checks are skipped; mandatory execution preflight is covered
    # independently below and in test_execution_preflight.py.
    assert "missing prerequisites" not in result.output


@pytest.mark.parametrize("skip_preflight", [False, True])
def test_submit_rechecks_execution_scope_before_persist_or_staging(
    monkeypatch: pytest.MonkeyPatch, mocker, tmp_path: Path, skip_preflight: bool
) -> None:
    from npa import execution_preflight
    from npa.orchestration.npa_workflow import first_run_state

    _mock_sky_bin_ok(monkeypatch)

    spec = tmp_path / "scope.yaml"
    spec.write_text(
        "apiVersion: npa.workflow/v0.0.1\n"
        "kind: Workflow\n"
        "metadata: {name: scope}\n"
        "config: {bucket: unit-output}\n"
        "resources: {cpu: {cloud: kubernetes, cpus: 1, memory: 1Gi}}\n"
        "initial: execute\n"
        "states:\n"
        "  execute:\n"
        "    resources: cpu\n"
        "    run: {shell: 'true'}\n"
        "    terminal: true\n",
        encoding="utf-8",
    )
    selected = execution_preflight.ExecutionTarget(
        project="unit",
        project_id="project-unit",
        tenant_id="tenant-unit",
        region="eu-west1",
        context="unit-context",
        output_uris=("s3://unit-output/results/",),
    )
    events = []

    def initial_preflight(*args, **kwargs):
        events.append("execution-target")
        return selected, {"execution_readiness": "pass"}

    def images(*args, **kwargs):
        events.append("images")
        return {}

    def missing_project(*args, **kwargs):
        events.append("execution-scope")
        return None

    monkeypatch.setattr(workflow_cli, "_execution_target_preflight", initial_preflight)
    monkeypatch.setattr(workflow_cli, "_preflight_submit_images", images)
    monkeypatch.setattr(
        workflow_cli, "_available_kube_contexts", lambda: ["unit-context"]
    )
    monkeypatch.setattr(workflow_cli, "_adopt_npa_kubeconfig", lambda context: True)
    monkeypatch.setattr(
        workflow_cli, "_verify_submit_controller_owner", lambda **kwargs: None
    )
    monkeypatch.setattr("npa.clients.nebius.get_project_identity", missing_project)
    scope = mocker.spy(execution_preflight, "verify_execution_scope")
    prepare = mocker.spy(first_run_state, "prepare_run")
    stage = mocker.patch("npa.cli.workbench.workflow._stage_npa_src_for_submit")
    runtime = mocker.patch("npa.cli.workbench.workflow._run_npa_workflow_runtime")
    launch = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")
    args = [
        "workbench",
        "workflow",
        "submit",
        str(spec),
        "--project",
        "unit",
        "--run-id",
        "scope-recheck",
        "--infra",
        "k8s/unit-context",
        "--sky-bin",
        "/bin/true",
        "--image",
        "ghcr.io/example/unit:dev",
        "--no-deploy-if-absent",
        "--no-stage-src",
    ]
    if skip_preflight:
        args.append("--skip-preflight")

    result = runner.invoke(app, args)

    assert result.exit_code == 1, result.output
    assert "execution preflight scope: selected project does not exist" in result.output
    assert events == ["execution-target", "images", "execution-scope"]
    scope.assert_called_once_with(selected)
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["persist"] is False
    stage.assert_not_called()
    runtime.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("requires_s3", [False, True])
def test_static_submit_prerequisites_never_probe_storage(
    monkeypatch: pytest.MonkeyPatch, mocker, requires_s3
) -> None:
    from npa.cli.workbench.workflow import _submit_prerequisites

    _mock_sky_bin_ok(monkeypatch)
    probe = mocker.patch("npa.clients.storage_validation.probe_storage_write")

    missing = _submit_prerequisites(
        {"bucket": "unit-output"} if requires_s3 else {},
        sky_bin="/bin/true",
        image="registry.example/npa-tool:v1",
        plan_only=False,
        requires_s3=requires_s3,
    )

    assert missing == []
    probe.assert_not_called()


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


def test_runtime_fetch_sonic_image_requires_staged_npa_source() -> None:
    """An image route is insufficient when the image omits the NPA CLI."""

    from npa.cli.workbench.workflow import _plan_requires_npa_source
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

    assert (
        _plan_requires_npa_source(
            SONIC_SPEC,
            run_id="sonic-runtime-fetch-source",
            assume_decision="",
            options=SkypilotRenderOptions(materialize_registry_secrets=False),
        )
        is True
    )


@pytest.mark.parametrize("overlay_origin", ["spec", "override", "environment"])
def test_pinned_b300_overlay_automatically_stages_and_reaches_worker(
    monkeypatch, mocker, overlay_origin
) -> None:
    """A digest override must not silently discard an explicit source overlay."""
    import yaml

    stage = mocker.patch("npa.orchestration.npa_workflow.src_staging.stage_npa_source")
    monkeypatch.setattr(workflow_cli, "_local_source_fingerprint", lambda: "a" * 64)
    monkeypatch.setattr(
        workflow_cli, "_resolve_submit_src_s3_uri_with_origin", lambda _: ("", "")
    )
    if overlay_origin == "environment":
        monkeypatch.setenv("NPA_SRC_OVERLAY", "1")
    spec = SPEC.parent / "flex-pi-b300-inference.yaml"
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(spec),
            "--run-id",
            "b300-overlay-plan",
            "--plan-only",
            "--no-deploy-if-absent",
            "--output-format",
            "json",
            "--image",
            f"cr.example.invalid/npa-flex-pi@sha256:{'b' * 64}",
            "--var",
            "bucket=example-bucket",
            *(["--var", "source_overlay=true"] if overlay_origin == "override" else []),
            *(
                ["--var", "source_overlay=false"]
                if overlay_origin == "environment"
                else []
            ),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source"]["status"] == "planned"
    task = [doc for doc in yaml.safe_load_all(payload["skypilot_yaml"]) if doc][-1]
    assert task["envs"]["NPA_SRC_OVERLAY"] == "1"
    assert task["envs"]["NPA_SRC_S3_URI"] == payload["source"]["uri"]
    assert task["envs"]["NPA_SRC_S3_URI"].endswith(f"/{'a' * 64}/")
    stage.assert_not_called()  # Plan-only remains read-only.


@pytest.mark.parametrize("baked", [False, True])
def test_pinned_b300_without_effective_overlay_needs_no_staged_source(baked) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

    assert not workflow_cli._plan_requires_npa_source(
        SPEC.parent / "flex-pi-b300-inference.yaml",
        run_id="b300-no-overlay",
        assume_decision="",
        config_overrides={
            "source_overlay": "true" if baked else "false",
            "require_baked_npa": "true" if baked else "false",
        },
        options=SkypilotRenderOptions(
            image_overrides={"*": f"cr.example.invalid/npa-flex-pi@sha256:{'b' * 64}"},
            materialize_registry_secrets=False,
        ),
    )


def test_preflight_images_accepts_the_same_config_vars_as_submit(mocker) -> None:
    """An empty canonical image input must be overridable before pull probes."""
    from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget

    digest_image = f"cr.example.invalid/npa@sha256:{'a' * 64}"
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )
    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="target-namespace"),
    )
    args = [
        "workbench",
        "workflow",
        "preflight-images",
        str(SIM2REAL_SPEC),
        "--assume-decision",
        "promote_checkpoint",
        "--infra",
        "k8s/target-context",
    ]
    for name in (
        "controller_image",
        "transfer_image",
        "envgen_image",
        "isaac_image",
        "viewer_image",
    ):
        args.extend(["--var", f"{name}={digest_image}"])

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    checked_images = checks.call_args.args[0]
    assert checked_images
    assert set(checked_images) == {digest_image}


def test_image_pull_execution_paths_preserve_each_rendered_target() -> None:
    requirements = {
        "vm": ImagePullRequirements(requires_operator=True),
        "kubernetes": ImagePullRequirements(requires_kubernetes=True),
        "mixed": ImagePullRequirements(
            requires_operator=True,
            requires_kubernetes=True,
        ),
    }

    operator, kubernetes = workflow_cli._image_pull_execution_paths(
        images=list(requirements),
        requirements=requirements,
    )

    assert operator == {"vm", "mixed"}
    assert kubernetes == {"kubernetes", "mixed"}


def _write_pull_authority_spec(
    tmp_path: Path,
    clouds: tuple[str, ...],
    *,
    parallel: bool = False,
    region: str = "",
) -> Path:
    image = f"registry.example.invalid/workflow@sha256:{'a' * 64}"
    resources = {}
    states = {}
    for index, cloud in enumerate(clouds):
        name = f"step-{index}"
        resources[name] = {
            "image": image,
            "kubernetes": {
                "pod_config": {
                    "spec": {
                        "serviceAccountName": name,
                        "imagePullSecrets": [{"name": f"pull-{index}"}],
                        "nodeSelector": {"test.example.invalid/pool": name},
                    }
                }
            },
        }
        if cloud:
            resources[name]["cloud"] = cloud
        if region:
            resources[name]["region"] = region
        states[name] = {"resources": name, "run": {"shell": "true"}}
        if not parallel:
            if index == len(clouds) - 1:
                states[name]["terminal"] = True
            else:
                states[name]["next"] = f"step-{index + 1}"
    initial = "step-0"
    if parallel:
        initial = "fan-out"
        states[initial] = {"parallel": list(states), "next": "done"}
        states["done"] = {"terminal": True}
    path = tmp_path / "pull-authorities.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "npa.workflow/v0.0.1",
                "kind": "Workflow",
                "metadata": {"name": "pull-authorities"},
                "resources": resources,
                "initial": initial,
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("cloud", ["", "nebius", "kubernetes"])
def test_single_task_infra_override_preserves_kubernetes_authority(
    tmp_path: Path, cloud: str
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

    spec = workflow_cli.load_spec(_write_pull_authority_spec(tmp_path, (cloud,)))
    images, requirements = workflow_cli._plan_preflight_image_requirements(
        spec,
        run_id="single",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
        assume_decision="",
        infra="k8s/target-context",
    )

    assert workflow_cli._image_pull_execution_paths(
        images=images, requirements=requirements
    ) == (set(), set(images))
    requirement = requirements[images[0]]
    assert requirement.service_account_names == ("step-0",)
    assert requirement.pull_secret_name_sets == (("pull-0",),)
    assert json.loads(requirement.pod_placement_specs[0]) == {
        "nodeSelector": {"test.example.invalid/pool": "step-0"}
    }


@pytest.mark.parametrize("parallel", [False, True])
def test_multitask_infra_keeps_declared_and_overridden_pull_authorities(
    tmp_path: Path, parallel: bool
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

    spec = workflow_cli.load_spec(
        _write_pull_authority_spec(
            tmp_path, ("nebius", "kubernetes"), parallel=parallel
        )
    )
    images, requirements = workflow_cli._plan_preflight_image_requirements(
        spec,
        run_id="multi",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
        assume_decision="",
        infra="k8s/target-context",
    )

    # Pinned SkyPilot ignores --infra for chain DAGs and JobGroups. Runtime
    # singleton waves may apply it, so neither authority can be discarded.
    assert workflow_cli._image_pull_execution_paths(
        images=images, requirements=requirements
    ) == (set(images), set(images))
    requirement = requirements[images[0]]
    assert set(requirement.service_account_names) == {"step-0", "step-1"}
    assert set(requirement.pull_secret_name_sets) == {("pull-0",), ("pull-1",)}


@pytest.mark.parametrize("parallel", [False, True])
def test_multitask_infra_cannot_resolve_an_ignored_cloud_override(
    tmp_path: Path, parallel: bool
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

    spec = workflow_cli.load_spec(
        _write_pull_authority_spec(tmp_path, ("", "kubernetes"), parallel=parallel)
    )
    images, requirements = workflow_cli._plan_preflight_image_requirements(
        spec,
        run_id="unresolved",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
        assume_decision="",
        infra="k8s/target-context",
    )

    assert requirements[images[0]].target_unresolved
    assert workflow_cli._image_pull_execution_paths(
        images=images, requirements=requirements
    ) == (set(), set())


def test_multitask_different_context_fails_before_registry_or_cluster_checks(
    tmp_path: Path, mocker
) -> None:
    path = _write_pull_authority_spec(
        tmp_path, ("kubernetes", "kubernetes"), region="declared-context"
    )
    pulls = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials"
    )
    target = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target"
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(path),
            "--infra",
            "k8s/other",
        ],
    )

    assert result.exit_code == 1
    assert "SkyPilot ignores resource overrides for multi-task YAML" in result.output
    pulls.assert_not_called()
    target.assert_not_called()


@pytest.mark.parametrize("infra_prefix", ["k8s", "kubernetes"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("declared_context", ["foo/bar", "foo"])
def test_multitask_context_with_slash_requires_exact_probe_target(
    tmp_path: Path,
    mocker,
    infra_prefix: str,
    parallel: bool,
    declared_context: str,
) -> None:
    from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget

    path = _write_pull_authority_spec(
        tmp_path,
        ("kubernetes", "kubernetes"),
        parallel=parallel,
        region=declared_context,
    )
    pulls = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    target = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="target-namespace"),
    )
    bootstrap = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(path),
            "--infra",
            f"{infra_prefix}/foo/bar",
        ],
    )

    if declared_context == "foo/bar":
        assert result.exit_code == 0, result.output
        assert target.call_args.kwargs["context"] == "foo/bar"
        assert pulls.call_args.kwargs["context"] == "foo/bar"
        assert bootstrap.call_args.kwargs["context"] == "foo/bar"
    else:
        assert result.exit_code == 1
        assert "different --infra context" in result.output
        target.assert_not_called()
        pulls.assert_not_called()
        bootstrap.assert_not_called()


@pytest.mark.parametrize("parallel", [False, True])
def test_multitask_submit_manifest_gate_cannot_skip_declared_vm_failure(
    tmp_path: Path, mocker, parallel: bool
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
    from npa.orchestration.skypilot.registry_preflight import ImagePullCheck

    path = _write_pull_authority_spec(
        tmp_path, ("nebius", "kubernetes"), parallel=parallel
    )
    pulls = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        side_effect=lambda images, **_kwargs: [
            ImagePullCheck(image=image, status="denied", detail="operator denied")
            for image in images
        ],
    )

    with pytest.raises(typer.Exit):
        workflow_cli._preflight_submit_image_manifests(
            path,
            options=SkypilotRenderOptions(materialize_registry_secrets=False),
            assume_decision="",
            enabled=True,
            infra="k8s/target-context",
        )

    pulls.assert_called_once()


def test_effective_pull_secret_sets_do_not_union_distinct_paths() -> None:
    requirements = {
        "image": ImagePullRequirements(
            requires_kubernetes=True,
            pull_secret_name_sets=(("path-a",), ("path-b",)),
        )
    }

    effective = workflow_cli._image_pull_secret_sets(
        images=["image"],
        requirements=requirements,
        kubernetes_images={"image"},
        inherited_pull_secrets=("global-primary", "global-secondary"),
    )

    assert effective == {
        "image": (
            ("path-a", "global-secondary"),
            ("path-b", "global-secondary"),
        )
    }


def test_effective_pull_paths_preserve_service_account_authority() -> None:
    requirements = {
        "image": ImagePullRequirements(
            requires_kubernetes=True,
            pull_secret_name_sets=(("shared",), ("shared",)),
            service_account_names=("service-account-a", "service-account-b"),
        )
    }

    effective = workflow_cli._image_pull_paths(
        images=["image"],
        requirements=requirements,
        kubernetes_images={"image"},
        inherited_service_account_name="base-service-account",
    )

    assert effective == {
        "image": (
            (("shared",), "service-account-a", "{}"),
            (("shared",), "service-account-b", "{}"),
        )
    }


def test_effective_pull_paths_merge_global_and_task_placement() -> None:
    requirements = {
        "image": ImagePullRequirements(
            requires_kubernetes=True,
            pull_secret_name_sets=(("shared",),),
            service_account_names=("service-account",),
            pod_placement_specs=(
                json.dumps(
                    {
                        "nodeSelector": {"nebius.com/node-group": "gpu"},
                        "runtimeClassName": "nvidia",
                    }
                ),
            ),
        )
    }

    effective = workflow_cli._image_pull_paths(
        images=["image"],
        requirements=requirements,
        kubernetes_images={"image"},
        inherited_service_account_name="base-service-account",
        inherited_pod_placement_json=json.dumps(
            {"nodeSelector": {"kubernetes.io/os": "linux"}}
        ),
    )

    assert effective == {
        "image": (
            (
                ("shared",),
                "service-account",
                (
                    '{"nodeSelector":{"kubernetes.io/os":"linux",'
                    '"nebius.com/node-group":"gpu"},'
                    '"runtimeClassName":"nvidia"}'
                ),
            ),
        )
    }


def test_unresolved_cloud_is_not_certified_as_operator_pull() -> None:
    requirements = {
        "image": ImagePullRequirements(
            target_unresolved=True,
        )
    }

    operator_images, kubernetes_images = workflow_cli._image_pull_execution_paths(
        images=["image"],
        requirements=requirements,
    )

    assert operator_images == set()
    assert kubernetes_images == set()


def test_effective_pull_secret_sets_reject_invalid_multi_entry_override() -> None:
    from npa.orchestration.skypilot.registry_preflight import RegistryPreflightError

    requirements = {
        "image": ImagePullRequirements(
            requires_kubernetes=True,
            pull_secret_name_sets=(("task-a", "task-b"),),
        )
    }

    with pytest.raises(RegistryPreflightError, match="exactly one entry"):
        workflow_cli._image_pull_secret_sets(
            images=["image"],
            requirements=requirements,
            kubernetes_images={"image"},
            inherited_pull_secrets=("global",),
        )


def test_effective_pull_secret_sets_reject_empty_task_override_with_base() -> None:
    from npa.orchestration.skypilot.registry_preflight import RegistryPreflightError

    requirements = {
        "image": ImagePullRequirements(
            requires_kubernetes=True,
            pull_secret_name_sets=((),),
        )
    }

    with pytest.raises(RegistryPreflightError):
        workflow_cli._image_pull_secret_sets(
            images=["image"],
            requirements=requirements,
            kubernetes_images={"image"},
            inherited_pull_secrets=("global",),
        )


def test_effective_pull_secret_sets_reject_task_override_after_empty_base() -> None:
    from npa.orchestration.skypilot.registry_preflight import RegistryPreflightError

    requirements = {
        "image": ImagePullRequirements(
            requires_kubernetes=True,
            pull_secret_name_sets=(("task",),),
        )
    }

    with pytest.raises(RegistryPreflightError):
        workflow_cli._image_pull_secret_sets(
            images=["image"],
            requirements=requirements,
            kubernetes_images={"image"},
            inherited_pull_secrets=(),
            inherited_pull_secrets_configured=True,
        )


def test_effective_pull_secret_sets_accept_initial_multi_entry_task_list() -> None:
    requirements = {
        "image": ImagePullRequirements(
            requires_kubernetes=True,
            pull_secret_name_sets=(("task-a", "task-b"),),
        )
    }

    effective = workflow_cli._image_pull_secret_sets(
        images=["image"],
        requirements=requirements,
        kubernetes_images={"image"},
    )

    assert effective == {"image": (("task-a", "task-b"),)}


@pytest.mark.parametrize(
    ("timeout_args", "pull_timeout", "bootstrap_timeout"),
    [
        ([], 1800, 1800),
        (["--image-pull-timeout-seconds", "0"], 0, 1800),
        (["--image-pull-timeout-seconds", "3600"], 3600, 1800),
        (["--image-bootstrap-timeout-seconds", "0"], 0, 0),
        (
            [
                "--image-pull-timeout-seconds",
                "0",
                "--image-bootstrap-timeout-seconds",
                "45",
            ],
            0,
            45,
        ),
    ],
)
def test_preflight_images_scopes_explicit_pull_secret_to_bootstrap(
    mocker, timeout_args, pull_timeout, bootstrap_timeout
) -> None:
    from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget

    digest_image = f"cr.example.invalid/npa@sha256:{'a' * 64}"
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    contracts = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )
    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="target-namespace"),
    )
    args = [
        "workbench",
        "workflow",
        "preflight-images",
        str(SIM2REAL_SPEC),
        "--assume-decision",
        "promote_checkpoint",
        "--image-pull-secret",
        "operator-registry",
        "--infra",
        "k8s/target-context",
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

    result = runner.invoke(app, args + timeout_args)

    assert result.exit_code == 0, result.output
    assert checks.call_args.kwargs["pull_secrets_by_image"] == {digest_image: ()}
    assert checks.call_args.kwargs["context"] == "target-context"
    assert checks.call_args.kwargs["namespace"] == "target-namespace"
    assert checks.call_args.kwargs["pull_secret_sets_by_image"] == {digest_image: ((),)}
    assert checks.call_args.kwargs["service_account_names_by_image"] == {
        digest_image: ("skypilot-service-account",)
    }
    assert checks.call_args.kwargs["operator_images"] == set()
    assert checks.call_args.kwargs["kubernetes_images"] == {digest_image}
    assert checks.call_args.kwargs["target_pull_timeout_seconds"] == pull_timeout
    assert (
        contracts.call_args.kwargs["observation_timeout_seconds"] == bootstrap_timeout
    )
    assert contracts.call_args.kwargs["pull_secrets_by_image"] == {
        digest_image: ("operator-registry",)
    }
    assert contracts.call_args.kwargs["service_accounts_by_image"] == {
        digest_image: "skypilot-service-account"
    }
    assert contracts.call_args.kwargs["context"] == "target-context"


@pytest.mark.parametrize("pull_timeout", [None, 0, 3600])
def test_submit_image_preflight_keeps_pull_and_bootstrap_deadlines_separate(
    mocker, pull_timeout
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

    image = "docker.io/library/alpine:3.22.0"
    mocker.patch.object(
        workflow_cli,
        "_plan_preflight_image_requirements",
        return_value=([image], {}),
    )
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    contracts = mocker.patch.object(
        workflow_cli, "_preflight_image_bootstrap_contracts", return_value=[]
    )
    workflow_cli._preflight_submit_images(
        SIM2REAL_SPEC,
        spec=SimpleNamespace(name="probe"),
        options=SkypilotRenderOptions(),
        assume_decision="",
        enabled=True,
        image_bootstrap_timeout_seconds=45,
        image_pull_timeout_seconds=pull_timeout,
    )
    assert checks.call_args.kwargs["target_pull_timeout_seconds"] == (
        45 if pull_timeout is None else pull_timeout
    )
    assert contracts.call_args.kwargs["observation_timeout_seconds"] == 45


def test_preflight_images_deduplicates_declared_and_explicit_pull_secret(
    mocker,
) -> None:
    from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget

    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="target-namespace"),
    )
    digest_image = f"cr.example.invalid/npa@sha256:{'a' * 64}"
    mocker.patch(
        "npa.cli.workbench.workflow._plan_preflight_image_requirements",
        return_value=(
            [digest_image],
            {
                digest_image: ImagePullRequirements(
                    requires_kubernetes=True,
                    pull_secret_name_sets=(("operator-registry",),),
                    service_account_names=(None,),
                    pod_placement_specs=("{}",),
                )
            },
        ),
    )
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(COSMOS3_SPEC),
            "--infra",
            "k8s/unit-context",
            "--registry",
            _OPERATOR_REGISTRY,
            "--image-pull-secret",
            "operator-registry",
        ],
    )

    assert result.exit_code == 0, result.output
    assert checks.call_args.kwargs["pull_secrets_by_image"] == {
        digest_image: ("operator-registry",)
    }


def test_preflight_images_covers_every_decision_branch(mocker) -> None:
    from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget

    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="target-namespace"),
    )
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(COSMOS3_SPEC),
            "--infra",
            "k8s/unit-context",
            "--registry",
            _OPERATOR_REGISTRY,
            "--image-pull-secret",
            "operator-registry",
        ],
    )

    assert result.exit_code == 0, result.output
    checked_images = checks.call_args.args[0]
    assert len(checked_images) == 5
    assert any("npa-cosmos-curate:" in image for image in checked_images)
    assert any("npa-fiftyone:" in image for image in checked_images)
    assert checks.call_args.kwargs["pull_secrets_by_image"] == {
        image: () for image in checked_images
    }


def test_preflight_images_reports_valid_empty_plan(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.workflow._plan_preflight_image_requirements",
        return_value=([], {}),
    )
    pulls = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials"
    )
    contracts = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts"
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(COSMOS3_SPEC),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == "images: none pinned by this spec\n"
    pulls.assert_not_called()
    contracts.assert_not_called()


@pytest.mark.parametrize("error_type", [ValueError, NpaWorkflowError])
def test_preflight_images_reports_planning_failure_before_pull_checks(
    mocker,
    error_type,
) -> None:
    mocker.patch(
        "npa.cli.workbench.workflow._plan_preflight_image_requirements",
        side_effect=error_type("synthetic complete-path planner failure"),
    )
    pulls = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials"
    )
    contracts = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts"
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(COSMOS3_SPEC),
        ],
    )

    assert result.exit_code == 1
    assert result.output == (
        "Error: image preflight planning failed: "
        "synthetic complete-path planner failure\n"
    )
    assert "images: none" not in result.output
    pulls.assert_not_called()
    contracts.assert_not_called()


def test_preflight_images_uses_selected_cluster_context_for_pull_authority(
    mocker,
) -> None:
    from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget

    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="target-namespace"),
    )
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    contracts = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(COSMOS3_SPEC),
            "--registry",
            _OPERATOR_REGISTRY,
            "--infra",
            "k8s/unit-context",
        ],
    )

    assert result.exit_code == 0, result.output
    assert checks.call_args.kwargs["context"] == "unit-context"
    assert contracts.call_args.kwargs["context"] == "unit-context"


def test_preflight_images_fails_on_branch_only_image(mocker) -> None:
    from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget

    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="target-namespace"),
    )
    from npa.orchestration.skypilot.registry_preflight import ImagePullCheck

    def branch_failure(images, **_kwargs):
        return [
            ImagePullCheck(
                image=image,
                status=("denied" if "npa-cosmos-curate:" in image else "ok"),
                detail=(
                    "synthetic branch-only pull failure"
                    if "npa-cosmos-curate:" in image
                    else ""
                ),
            )
            for image in images
        ]

    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        side_effect=branch_failure,
    )
    contracts = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts"
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(COSMOS3_SPEC),
            "--registry",
            _OPERATOR_REGISTRY,
        ],
    )

    assert result.exit_code == 1
    assert "npa-cosmos-curate:" in result.output
    assert "synthetic branch-only pull failure" in result.output
    assert any("npa-cosmos-curate:" in image for image in checks.call_args.args[0])
    contracts.assert_not_called()


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
_PINNED_IMAGE = "registry.example/reg/npa-lerobot:tag"


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
    / "workflows"
    / "testing"
    / "adversarial-scenario-hardening.yaml"
)


@pytest.fixture
def image_selector_boundaries(mocker):
    targets = (
        "npa.cli.workbench.workflow._refuse_dedicated_live_gate_execution",
        "npa.orchestration.npa_workflow.robotwin_preflight.prepare_live_submit",
        "npa.orchestration.npa_workflow.first_run_state.prepare_run",
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        "npa.cli.workbench.workflow._preflight_submit_images",
        "npa.cli.workbench.workflow._stage_npa_src_for_submit",
        "npa.orchestration.npa_workflow.deploy.ensure_infra_present",
        "npa.orchestration.skypilot.workflow.submit_workflow",
        "npa.cli.workbench.workflow._run_npa_workflow_runtime",
    )
    boundaries = []
    for target in targets:
        boundary = mocker.patch(target, side_effect=AssertionError(target))
        boundaries.append(boundary)
    return boundaries


@pytest.mark.parametrize("preflight", ["--preflight-images", "--no-preflight-images"])
@pytest.mark.parametrize("dedicated_live_gate", [False, True])
@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("workbench.scenario_gen.generat", "matched no workflow toolRef"),
        ("workbench.scenario_gen.genreate", "matched no workflow toolRef"),
        ("workbench.*", "Use TOOL_REF=IMAGE"),
    ],
)
def test_submit_rejects_image_selector_before_mutations(
    image_selector_boundaries, mocker, preflight, selector, message, dedicated_live_gate
):
    mocker.patch(
        "npa.cli.workbench.workflow._is_dedicated_live_gate_spec",
        return_value=dedicated_live_gate,
    )
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(HARDENING_SPEC),
            "--run-id",
            "invalid-image-selector",
            "--runtime",
            "--deploy-if-absent",
            "--image-override",
            f"{selector}=cr.example/custom:1",
            preflight,
        ],
    )

    assert result.exit_code == 1
    assert message in result.output
    for boundary in image_selector_boundaries:
        boundary.assert_not_called()


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
    prerequisites = mocker.patch(
        "npa.cli.workbench.workflow._submit_prerequisites",
        side_effect=AssertionError("submit prerequisites reached after quota blocker"),
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
    prerequisites.assert_not_called()
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
    mocker.patch("npa.cli.workbench.workflow._preflight_submit_image_manifests")
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


@pytest.mark.parametrize("pull_timeout", [0, 3600])
def test_submit_lets_a_deploy_if_absent_spec_provision_its_own_context(
    monkeypatch, tmp_path, mocker, pull_timeout
) -> None:
    """The preflight used to reject the context that --deploy-if-absent creates."""
    monkeypatch.setenv("KUBECONFIG", str(_write_kubeconfig(tmp_path, "other-ctx")))
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://real-bucket/npa-src/npa")
    _mock_sky_bin_ok(monkeypatch)
    planned_targets = []
    provisioned_targets = []
    phases: list[str] = []

    def plan(targets, *, mutation):
        assert mutation is True
        planned_targets.extend(targets)
        return {"submit-context": mocker.Mock()}

    def ensure(targets, **_kwargs):
        phases.append("provision")
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
    mocker.patch(
        "npa.cli.workbench.workflow._preflight_submit_image_manifests",
        side_effect=lambda *args, **kwargs: phases.append("manifest"),
    )
    target_preflight = mocker.patch(
        "npa.cli.workbench.workflow._preflight_submit_images",
        side_effect=lambda *args, **kwargs: phases.append("target") or {},
    )
    mocker.patch("npa.cli.workbench.workflow._adopt_npa_kubeconfig", return_value=True)
    mocker.patch("npa.cli.workbench.workflow._verify_submit_controller_owner")
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
            "--image-pull-timeout-seconds",
            str(pull_timeout),
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
    assert phases == ["manifest", "provision", "target"]
    target_preflight.assert_called_once()
    assert (
        target_preflight.call_args.kwargs["image_pull_timeout_seconds"] == pull_timeout
    )
