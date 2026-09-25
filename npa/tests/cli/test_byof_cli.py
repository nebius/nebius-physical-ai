"""CLI coverage for npa workbench byof."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import errno
import json
import os
import re
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
import typer
import yaml
from typer.main import get_command
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import byof as byof_cli
from npa.cli.workbench.byof import build_byof_argv
from npa.sdk.workbench import byof as byof_sdk
from npa.orchestration.npa_workflow.robotwin_preflight import (
    CUSTOMER_TERMS,
    CUSTOMER_USE_SCOPE,
    MATERIALIZED_CUSTOMER_ENTITLEMENT_ENV,
    MATERIALIZED_KUBECONFIG_ENV,
    MATERIALIZED_SKYPILOT_CONFIG_ENV,
    PUBLIC_CONTEXT_ENV,
    RUNTIME_LOCK_SHA256,
    TRANSPORT_CONTEXT_ENV,
    encode_transport,
    load_runtime_authorization,
)
from npa.orchestration.npa_workflow import robotwin_preflight

runner = CliRunner()


def test_byof_runner_resolves_from_staged_npa_source(
    tmp_path: Path, monkeypatch
) -> None:
    staged_module = tmp_path / "src/npa/cli/workbench/byof.py"
    staged_module.parent.mkdir(parents=True)
    staged_module.touch()
    staged_runner = tmp_path / "scripts/run_byof_repo.py"
    staged_runner.parent.mkdir()
    staged_runner.touch()
    monkeypatch.delenv("NPA_REPO_ROOT", raising=False)
    monkeypatch.setattr(byof_cli, "__file__", str(staged_module))

    assert byof_cli._script_path() == staged_runner


def test_byof_registered_in_workbench_help() -> None:
    result = runner.invoke(app, ["workbench", "--help"])
    assert result.exit_code == 0
    assert re.search(r"│\s+byof\s+", result.output)


def test_byof_run_help() -> None:
    result = runner.invoke(
        app,
        ["workbench", "byof", "run", "--help"],
        terminal_width=200,
    )
    assert result.exit_code == 0
    assert "--repo-url" in result.output
    assert "--workload" in result.output
    assert "--base-profile" in result.output
    assert "--repo-auth" in result.output
    assert "--repo-token-env" in result.output
    byof = get_command(app).commands["workbench"].commands["byof"]
    run_command = byof.commands["run"]
    runtime_context_options = [
        parameter
        for parameter in run_command.params
        if "--runtime-context-env" in getattr(parameter, "opts", ())
    ]
    assert len(runtime_context_options) == 1
    assert runtime_context_options[0].name == "runtime_context_env"


def test_byof_run_dry_run_json() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "byof",
            "run",
            "--repo-url",
            "https://github.com/example/repo.git",
            "--repo-ref",
            "main",
            "--workload",
            "container-verify",
            "--dry-run",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["script"] == "npa/scripts/run_byof_repo.py"
    assert "--repo-url" in payload["argv"]
    assert "https://github.com/example/repo.git" in payload["argv"]
    assert payload["ladder"] == "docs/architecture/oss-onboarding-ladder.md"


def test_byof_ladder_and_status() -> None:
    ladder = runner.invoke(app, ["workbench", "byof", "ladder", "--output", "json"])
    assert ladder.exit_code == 0
    ladder_payload = json.loads(ladder.output)
    assert ladder_payload["tiers"][0]["tier"] == 0

    status = runner.invoke(app, ["workbench", "byof", "status", "--output", "json"])
    assert status.exit_code == 0
    status_payload = json.loads(status.output)
    assert status_payload["cli"] == "npa workbench byof"
    assert status_payload["sdk"] == "npa.sdk.workbench.byof"
    assert "workbench.byof.repo" in status_payload["tool_refs"]


def test_build_byof_argv_and_sdk_plan() -> None:
    argv = build_byof_argv(
        repo_url="https://github.com/example/repo.git",
        repo_ref="main",
        workload="container-verify",
        skip_run=True,
    )
    assert "--skip-run" in argv
    assert (
        byof_sdk.plan_argv(
            repo_url="https://github.com/example/repo.git",
            repo_ref="main",
            workload="container-verify",
            skip_run=True,
        )
        == argv
    )


def test_private_byof_sdk_plan_carries_only_token_variable_name() -> None:
    argv = byof_sdk.plan_argv(
        repo_url="https://github.com/example/private.git",
        repo_ref="main",
        repo_auth="github",
        repo_token_env="NPA_BYOF_GITHUB_TOKEN",
        skip_run=True,
    )

    assert argv[argv.index("--repo-auth") + 1] == "github"
    assert argv[argv.index("--repo-token-env") + 1] == "NPA_BYOF_GITHUB_TOKEN"
    assert "secret-canary" not in " ".join(argv)


def test_robotwin_dry_run_carries_only_runtime_context_variable_name(
    monkeypatch,
) -> None:
    variable = "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT"
    secret = "runtime-context-secret-canary"
    monkeypatch.setenv(variable, secret)
    result = runner.invoke(
        app,
        [
            "workbench",
            "byof",
            "run",
            "--repo-url",
            "https://github.com/RoboTwin-Platform/RoboTwin.git",
            "--solution-name",
            "robotwin",
            "--runtime-context-env",
            variable,
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    argv = json.loads(result.output)["argv"]
    assert argv[argv.index("--runtime-context-env") + 1] == variable
    assert secret not in " ".join(argv)


@pytest.mark.parametrize(
    "identity_args",
    [
        [
            "--repo-url",
            "https://github.com:443/RoboTwin-Platform/RoboTwin.git/",
        ],
        [
            "--repo-url",
            "https://github.com/example/not-robotwin.git",
            "--image",
            "registry.invalid/private/npa-robotwin:mutable",
        ],
    ],
)
def test_robotwin_equivalent_or_mutable_direct_cli_refuses_before_loading_runner(
    monkeypatch: pytest.MonkeyPatch, identity_args: list[str]
) -> None:
    monkeypatch.delenv(TRANSPORT_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(
        byof_cli,
        "_load_runner",
        lambda: pytest.fail("direct RoboTwin CLI loaded the live BYOF runner"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "byof",
            "run",
            *identity_args,
            "--runtime-context-env",
            PUBLIC_CONTEXT_ENV,
        ],
    )

    assert result.exit_code == 2
    assert "only through normal" in result.output


def _robotwin_transport_fixture(
    tmp_path: Path,
) -> tuple[list[str], str, object]:
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3] / "workflows/testing/byof-robotwin.yaml"
        ).read_text(encoding="utf-8")
    )
    config = workflow["config"]
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
    kubeconfig.chmod(0o600)
    skypilot.chmod(0o600)
    context = tmp_path / "runtime-context.json"
    context.write_text(
        json.dumps(
            {
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
                "project": "robotwin-project-canary",
                "nebius_profile": "robotwin-profile",
                "kubeconfig": str(kubeconfig),
                "kubernetes_context": "robotwin-context",
                "skypilot_config_path": str(skypilot),
                "bucket": "robotwin-bucket-canary",
                "output_root": "s3://robotwin-bucket-canary/output",
                "run_id": "robotwin-run-canary",
            }
        ),
        encoding="utf-8",
    )
    context.chmod(0o600)
    materialized_dir = tmp_path / "materialized-config"
    materialized_dir.mkdir(mode=0o700)
    materialized_kubeconfig = materialized_dir / "kubeconfig.yaml"
    materialized_skypilot = materialized_dir / "skypilot.yaml"
    materialized_kubeconfig.write_bytes(kubeconfig.read_bytes())
    materialized_skypilot.write_bytes(skypilot.read_bytes())
    materialized_kubeconfig.chmod(0o600)
    materialized_skypilot.chmod(0o600)
    assertion = SimpleNamespace(
        issuer="https://customer-auth.example.invalid",
        customer_scope_id="robotwin-customer-canary",
        run_id="robotwin-run-canary",
        runtime_manifest_sha256=RUNTIME_LOCK_SHA256,
        issued_at="2026-01-01T00:00:00Z",
        expires_at="2099-01-01T00:00:00Z",
        decision="accepted",
        intended_activity=CUSTOMER_USE_SCOPE,
        terms=list(CUSTOMER_TERMS),
        assertion_id="assertion-canary-0001",
        nonce="nonce-canary-00000001",
    )
    authorization = load_runtime_authorization(
        {
            PUBLIC_CONTEXT_ENV: str(context),
            MATERIALIZED_KUBECONFIG_ENV: str(materialized_kubeconfig),
            MATERIALIZED_SKYPILOT_CONFIG_ENV: str(materialized_skypilot),
        },
        customer_authorization_boundary=SimpleNamespace(
            trusted_issuer="https://customer-auth.example.invalid",
            consume_once=lambda _request: assertion,
        ),
    )
    argv = build_byof_argv(
        repo_url=config["repo_url"],
        repo_ref=config["repo_ref"],
        repo_auth=config["repo_auth"],
        repo_token_env=config["repo_token_env"],
        base_profile=config["base_profile"],
        base_image=config["base_image"],
        workload=config["workload"],
        build_command=config["build_command"],
        smoke_command=config["smoke_command"],
        solution_name=config["solution_name"],
        capability_name=config["capability_name"],
        smoke_artifact_name=config["smoke_artifact_name"],
        runtime_context_env=config["runtime_context_env"],
        task=config["task"],
        iterations=config["iterations"],
        num_envs=config["num_envs"],
        num_demos=config["num_demos"],
        yaml_path=config["resource_profile_yaml"],
        output_root="s3://example-bucket/oss-solutions/robotwin",
        wait_timeout=config["wait_timeout"],
        poll_interval=config["poll_interval"],
    )
    return argv, encode_transport(authorization), authorization


@pytest.mark.parametrize("failure", [None, RuntimeError, SystemExit])
def test_robotwin_worker_transport_materializes_owner_only_and_always_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[BaseException] | None,
) -> None:
    runner_module = byof_cli._load_runner()
    argv, transport, fixture_authorization = _robotwin_transport_fixture(tmp_path)
    monkeypatch.setattr(
        robotwin_preflight,
        "decode_transport",
        lambda _value: fixture_authorization,
    )
    monkeypatch.setattr(
        robotwin_preflight,
        "load_runtime_authorization",
        lambda environ, **_kwargs: replace(
            fixture_authorization,
            kubeconfig_source=environ[MATERIALIZED_KUBECONFIG_ENV],
            skypilot_config_source=environ[MATERIALIZED_SKYPILOT_CONFIG_ENV],
        ),
    )
    monkeypatch.delenv(PUBLIC_CONTEXT_ENV, raising=False)
    monkeypatch.setenv(TRANSPORT_CONTEXT_ENV, transport)
    materialized_paths: list[Path] = []

    expectation = pytest.raises(failure) if failure is not None else nullcontext()
    with expectation:
        with byof_cli._robotwin_runtime_materialization(runner_module, argv) as (
            authorization,
            runtime_environment,
        ):
            assert runtime_environment is not None
            assert os.environ[TRANSPORT_CONTEXT_ENV] == transport
            assert PUBLIC_CONTEXT_ENV not in os.environ
            context = Path(runtime_environment[PUBLIC_CONTEXT_ENV])
            authorization_receipt = Path(
                runtime_environment[MATERIALIZED_CUSTOMER_ENTITLEMENT_ENV]
            )
            kubeconfig = Path(runtime_environment[MATERIALIZED_KUBECONFIG_ENV])
            skypilot = Path(runtime_environment[MATERIALIZED_SKYPILOT_CONFIG_ENV])
            materialized_paths.extend(
                (context, authorization_receipt, kubeconfig, skypilot)
            )
            assert authorization is not None
            assert authorization.kubeconfig_source == str(kubeconfig)
            assert authorization.skypilot_config_source == str(skypilot)
            assert stat.S_IMODE(context.parent.stat().st_mode) == 0o700
            assert all(
                stat.S_IMODE(path.stat().st_mode) == 0o600
                for path in materialized_paths
            )
            if failure is not None:
                raise failure("fixed-test-failure")

    assert all(not path.exists() for path in materialized_paths)
    assert os.environ[TRANSPORT_CONTEXT_ENV] == transport
    assert PUBLIC_CONTEXT_ENV not in os.environ
    assert MATERIALIZED_CUSTOMER_ENTITLEMENT_ENV not in os.environ


def test_robotwin_materialization_rollback_records_partial_write_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _argv, _transport, fixture_authorization = _robotwin_transport_fixture(tmp_path)
    materialized_dir = tmp_path / "materialized-rollback"
    materialized_dir.mkdir(mode=0o700)
    real_write_owner_file = robotwin_preflight.write_owner_file
    calls = 0

    def fail_after_first_write(
        path: Path, raw: bytes, *, directory_fd: int | None = None
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "injected rollback boundary")
        real_write_owner_file(path, raw, directory_fd=directory_fd)

    monkeypatch.setattr(robotwin_preflight, "write_owner_file", fail_after_first_write)

    with pytest.raises(OSError, match="injected rollback boundary") as failure:
        robotwin_preflight._materialize_authorization(
            fixture_authorization, materialized_dir
        )

    recovery = failure.value.recovery_context
    assert calls == 2
    assert recovery.cleanup_outcomes == (("runtime-context.json", "removed"),)
    assert recovery.residual_names == ()
    assert recovery.directory_fsync == "synced"
    assert not any(materialized_dir.iterdir())


def test_robotwin_worker_materialization_never_mutates_process_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_module = byof_cli._load_runner()
    argv, transport, fixture_authorization = _robotwin_transport_fixture(tmp_path)
    monkeypatch.setattr(
        robotwin_preflight,
        "decode_transport",
        lambda _value: fixture_authorization,
    )
    monkeypatch.setattr(
        robotwin_preflight,
        "load_runtime_authorization",
        lambda environ, **_kwargs: replace(
            fixture_authorization,
            kubeconfig_source=environ[MATERIALIZED_KUBECONFIG_ENV],
            skypilot_config_source=environ[MATERIALIZED_SKYPILOT_CONFIG_ENV],
        ),
    )
    monkeypatch.setenv(TRANSPORT_CONTEXT_ENV, transport)
    before = dict(os.environ)

    with byof_cli._robotwin_runtime_materialization(runner_module, argv) as (
        authorization,
        runtime_environment,
    ):
        assert authorization is not None
        assert runtime_environment is not None
        assert dict(os.environ) == before
        assert runtime_environment is not os.environ

    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "failure_boundary", ["materialize_transport", "load_runtime_authorization"]
)
def test_robotwin_worker_transport_failure_detaches_private_exception_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_boundary: str,
) -> None:
    runner_module = byof_cli._load_runner()
    argv, transport, _authorization = _robotwin_transport_fixture(tmp_path)
    private_canary = "owner-only-worker-context-canary"
    private_digest = "b" * 64

    def fail_privately(*_args: object, **_kwargs: object) -> None:
        try:
            raise RuntimeError(private_canary)
        except RuntimeError as cause:
            raise robotwin_preflight.RobotwinPreflightError(
                "worker-context-invalid", private_digest
            ) from cause

    monkeypatch.delenv(PUBLIC_CONTEXT_ENV, raising=False)
    monkeypatch.setenv(TRANSPORT_CONTEXT_ENV, transport)
    monkeypatch.setattr(robotwin_preflight, failure_boundary, fail_privately)

    with pytest.raises(typer.BadParameter) as failure:
        with byof_cli._robotwin_runtime_materialization(runner_module, argv):
            pytest.fail("invalid worker authorization reached the BYOF runner")

    assert str(failure.value) == "RoboTwin worker authorization refused"
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert private_canary not in str(failure.value)
    assert private_digest not in str(failure.value)
    assert os.environ[TRANSPORT_CONTEXT_ENV] == transport
    assert PUBLIC_CONTEXT_ENV not in os.environ


@pytest.mark.parametrize(
    "failure_boundary", ["materialize_transport", "load_runtime_authorization"]
)
@pytest.mark.parametrize("terminal_width", [40, 100, 200])
def test_robotwin_worker_transport_failure_keeps_private_details_out_of_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_boundary: str,
    terminal_width: int,
) -> None:
    argv, transport, _authorization = _robotwin_transport_fixture(tmp_path)
    private_canary = "owner-only-worker-cli-canary"
    private_digest = "c" * 64

    def fail_privately(*_args: object, **_kwargs: object) -> None:
        try:
            raise RuntimeError(private_canary)
        except RuntimeError as cause:
            raise robotwin_preflight.RobotwinPreflightError(
                "worker-context-invalid", private_digest
            ) from cause

    monkeypatch.delenv(PUBLIC_CONTEXT_ENV, raising=False)
    monkeypatch.setenv(TRANSPORT_CONTEXT_ENV, transport)
    monkeypatch.setattr(robotwin_preflight, failure_boundary, fail_privately)

    monkeypatch.setenv("COLUMNS", str(terminal_width))
    result = runner.invoke(
        app, ["workbench", "byof", "run", *argv], terminal_width=terminal_width
    )

    # Rich wraps prose inside a bordered error panel; retain the whole message
    # contract without depending on which words share a rendered line.
    diagnostic = " ".join(result.output.replace("│", " ").split())
    compact_diagnostic = "".join(diagnostic.split())
    assert result.exit_code == 2
    assert "Phase A worker bridge is disabled" in diagnostic
    assert private_canary not in result.output
    assert private_digest not in result.output
    assert private_canary not in compact_diagnostic
    assert private_digest not in compact_diagnostic
    assert os.environ[TRANSPORT_CONTEXT_ENV] == transport
    assert PUBLIC_CONTEXT_ENV not in os.environ


@pytest.mark.parametrize("terminal_width", [40, 100, 200])
def test_robotwin_caller_supplied_transport_cannot_activate_internal_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal_width: int,
) -> None:
    real_runner = byof_cli._load_runner()
    argv, transport, _authorization = _robotwin_transport_fixture(tmp_path)
    observed: dict[str, object] = {}

    def run_authorized(received: list[str], *, authorization: object) -> int:
        observed["argv"] = received
        observed["authorization"] = authorization
        return 0

    monkeypatch.setenv(TRANSPORT_CONTEXT_ENV, transport)
    monkeypatch.delenv(PUBLIC_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(
        byof_cli,
        "_load_runner",
        lambda: SimpleNamespace(
            _parse_args=real_runner._parse_args,
            main=lambda *_args: pytest.fail("worker used the direct script boundary"),
            _run_authorized_robotwin=run_authorized,
        ),
    )

    monkeypatch.setenv("COLUMNS", str(terminal_width))
    result = runner.invoke(
        app, ["workbench", "byof", "run", *argv], terminal_width=terminal_width
    )

    assert result.exit_code == 2
    diagnostic = " ".join(result.output.replace("│", " ").split())
    assert "Phase A worker bridge is disabled" in diagnostic
    assert observed == {}


def test_robotwin_internal_cli_runner_signature_passes_only_validated_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner_module = byof_cli._load_runner()
    argv, transport, authorization = _robotwin_transport_fixture(tmp_path)
    observed: dict[str, object] = {}
    monkeypatch.setenv("UNBOUND_PRIVATE_OVERRIDE", "ambient-private-canary")
    monkeypatch.setattr(
        runner_module, "require_runtime_lock_complete", lambda value: value
    )

    def run_byof(args, *, authorization, **kwargs):
        observed.update(args=args, authorization=authorization, kwargs=kwargs)
        return 0

    monkeypatch.setattr(runner_module, "_run_byof", run_byof)

    assert (
        runner_module._run_authorized_robotwin(
            argv,
            authorization=authorization,
        )
        == 0
    )
    assert observed["authorization"] is authorization
    assert "environment" not in observed["kwargs"]
    assert "ambient-private-canary" not in json.dumps(
        observed["kwargs"], default=str, sort_keys=True
    )
